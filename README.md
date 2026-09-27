## Dataset

The dataset used in this project is the [Coffee Sales dataset](https://www.kaggle.com/datasets/ihelon/coffee-sales), available on Kaggle.

The raw CSV files are not included in this repository because they are downloaded directly from Kaggle. To reproduce the project locally, follow the steps below.

### 1. Install dependecies

The required packages can be installed either from the terminal using Bash or directly from a Jupyter Notebook.

Using Bash:

```bash
pip install kagglehub python-dotenv
```

Using Jupyter Notebook:

```python
!pip install kagglehub python-dotenv
```

### 2. Configure Kaggle API credentials

Create a Kaggle API token in your Kaggle account settings. Then, create a `.env` file in the root of the project with the following variables:

```env
KAGGLE_API_TOKEN=your_kaggle_api_token
BASE_PATH_DATASET=path/to/your/local/data/folder
```

### 3. Download the dataset

In your jupyter notebook, execute a cell with the following code:

```python
from dotenv import load_dotenv
from pathlib import Path
import os
import shutil
import pandas as pd
import kagglehub

load_dotenv()

os.environ["KAGGLE_API_TOKEN"] = os.getenv("KAGGLE_API_TOKEN")

base_path_dataset = Path(os.getenv("BASE_PATH_DATASET"))
base_path_dataset.mkdir(parents=True, exist_ok=True)

kaggle_path = Path(kagglehub.dataset_download("ihelon/coffee-sales"))

for file_path in kaggle_path.iterdir():
    if file_path.is_file():
        shutil.copy2(file_path, base_path_dataset / file_path.name)

csv_files = list(base_path_dataset.glob("*.csv"))
df = pd.read_csv(csv_files[0])
```
After running this code, the dataset will be available locally in the folder defined by `BASE_PATH_DATASET`.

### Gitignore recommendation

To avoid exposing credentials or uploading raw data files to GitHub, make sure the following entries are included in `.gitignore`:

```gitignore
.env
data/
```