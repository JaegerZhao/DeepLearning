import torch
from torch import nn
import torchvision
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Pretrained ImageNet ResNet-101
        # Remove linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune=True)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: boolean
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class DecoderWithRNN(nn.Module):
    def __init__(self, cfg, encoder_dim=14*14*2048):
        super(DecoderWithRNN, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        # To Do: define some layers for decoder with RNN
        # 嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        # LSTMCell 层
        self.decode_step = nn.LSTMCell(self.decoder_dim, self.decoder_dim)
        # 线性层
        self.init = nn.Linear(self.encoder_dim, self.decoder_dim)
        # 批量归一化层
        self.bn = nn.BatchNorm1d(self.decoder_dim)
        # 全连接层
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)
        # Dropout 层 
        self.dropout = nn.Dropout(self.dropout)

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.reshape(batch_size, -1)
        vocab_size = self.vocab_size

        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # Create tensors to hold word prediction scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        # Initialize LSTM state
        init_input = self.bn(self.init(encoder_out))
        h, c = self.decode_step(init_input)  # (batch_size, decoder_dim)

        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, generate a new word in the decoder with the previous word embedding
        for t in range(max(decode_lengths)):
            num = sum([1 if l > t else 0 for l in decode_lengths])
            preds, h, c = self.one_step(embeddings[:num,t,:], h[:num], c[:num])
            predictions[:num,t,:] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

    def one_step(self, embeddings, h, c):
        # To Do: Implement the one time decode step for forward pass 
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return hidden state and cell state: h, c
        output, (h, c) = self.decode_step(embeddings, (h, c))
        preds = self.fc(self.dropout(h))
        return preds, h, c

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # To Do: you need to define some layers for attention module
        # Hint: Firstly, define linear layers to transform encoded tensor
        # and decoder's output tensor to attention dim; Secondly, define
        # attention linear layer to calculate values to be softmax-ed;
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # To Do: Implement the forward pass for attention module
        # Hint: follow the equation 
        # "e = f_att(encoder_out, decoder_hidden)"
        # "alpha = softmax(e)"
        # "z = alpha * encoder_out"
        # Compute the attention scores with the function f_att
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        e = self.att(F.relu(att1 + att2)).squeeze(2)
        alpha = F.softmax(e)
        z = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return z, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder with Attention.
    """

    def __init__(self, cfg, encoder_dim=2048):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        # To Do: define some layers for decoder with attention
        self.attention = Attention(encoder_dim, self.decoder_dim, self.attention_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim)
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # Initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, decode by attention-weighing the encoder's output based 
        # on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
          num = sum([1 if l > t else 0 for l in decode_lengths])
          preds, alpha, h, c = self.one_step(embeddings[:num, t, :], encoder_out[:num], h[:num], c[:num])
          predictions[:num,t,:] = preds
          alphas[:num,t,:] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def one_step(self, embeddings, encoder_out, h, c):
        # To Do: Implement the one time decode step for forward pass
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return attention weight: alpha
        # return hidden state and cell state: h, c
        z, alpha = self.attention(encoder_out, h)
        b = F.sigmoid(self.beta(h))
        z = b * z
        h, c = self.decode_step(torch.cat([embeddings,z], dim=1),(h,c))
        preds = self.fc(self.dropout(h))

        return preds, alpha, h, c