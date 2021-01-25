# QAnT : image Quality Assessment aNd dicom Tags extraction

QAnT extracts no-reference IQMs (image quality metrics) representing noise/information measurements as well as DICOM 
metadata (Tags). 

The initial goal of this project was to extract IQMs and dicom tags from structural MRI images in order to automatically
determine a site/scanners effect of the images.

## Installation
### 1. Create a [conda](https://docs.conda.io/en/latest/) environment (recommended)
```
ENVNAME="QAnT"
conda create -n $ENVNAME python==3.7.7 -y
conda activate $ENVNAME
```
### 2. Install repository
#### Method 1: Github Master Branch
```
pip install git+https://github.com/Alxaline/QAnT.git
```
#### Method 2: Development Installation
```
git clone https://github.com/Alxaline/QAnT.git
cd QAnT
pip install -e .
```


## Usage

### Extraction

This tool takes datasets in the file formats (.dcm, .nii, .nii.gz) as the input. 
To parse DICOM files, the script need to have dicom series in an independent folder, i.e. a unique folder
for a volume with all .dcm slices inside. 

You need to provide a parameter file for extraction. An example is available in QAnT/example_parameters/default_parameters.yaml

The tool is multi-process in order to speed up the extraction process.

You can directly use the cli:
```
qant-extractor [-h] -i INPUT_DIR [INPUT_DIR ...] -o OUTPUT_FILEPATH [-p PARAM] [-j N] [-v]
```

or in python mode:
```
usage: python -m QAnT.extractor [-h] -i INPUT_DIR [INPUT_DIR ...] -o OUTPUT_FILEPATH
                    [-p PARAM] [-j N] [-v]

QAnT: image Quality Assessment and dicom Tags extraction

optional arguments:
  -h, --help            show this help message and exit

Required:
  -i INPUT_DIR [INPUT_DIR ...], --input_dir INPUT_DIR [INPUT_DIR ...]
                        Input directories path with DICOM files to be parsed.
                        Can be a list of directory
  -o OUTPUT_FILEPATH, --output_filepath OUTPUT_FILEPATH
                        Output filepath for saving the content in csv files.
                        Need to have the .csv extensions
  -p PARAM, --param PARAM
                        Parameter file containing the settings to be used in
                        extraction. If not provided use default setting.
  --inclusion_keywords INCLUSION_KEYWORDS [INCLUSION_KEYWORDS ...]
                        Inclusion keywords to parse files. fnmatch style, i.e
                        ['a*', 'b*']
  --exclusion_keywords EXCLUSION_KEYWORDS [EXCLUSION_KEYWORDS ...]
                        Exclusion keywords to parse files. fnmatch style, i.e
                        ['a*', 'b*']

Options:
  -j N, --n_jobs N      Specifies the number of threads to use for parallel
                        processing (default: all)
  -v, --verbosity       increase output verbosity (e.g., -vv is more than -v)
```

### Visualize

You can visualize results csv file in the application interface.

You can directly use the cli:
```
qant-interface 
```

or in python mode:
```
usage: python -m QAnT.interface 
```                    

# TODO: screen records

