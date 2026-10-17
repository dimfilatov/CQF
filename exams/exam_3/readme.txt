Instructions to run the full pipeline:

1. create a conda environment:
conda create -n my_env python=3.10

2. activate conda environment:
conda activate my_env

3. change directory to the root folder with exam files

4. install required dependencies:
pip install -r requirements.txt

5a. run exam_3.ipynb from jupyter notebook (contains more plots and tables than main.py). Exam pdf file was generated based on this notebook. In this file main.py is called at runtime.

5. Or, alternatively run full pipeline from the root (some plots and tables are only present in notebook):
python main.py
