from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/home/mw/project/coco2014/dataset_coco.json',   # downloaded from http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
                       image_folder='/home/mw/project/coco2014/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/home/mw/work/work_dir/coco/',
                       max_len=50)
