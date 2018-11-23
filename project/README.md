# Description of the dataset

The dataset is composed of two files:

- `en_docs_clean.csv`
- `pt_docs_clean.csv`

The first contains manifestos from the United Kingdom. The second contains
manifestos from Portugal. You can use one of them or both in your system.

Each **line** in a file contains the following columns:

- **text**: one segment of a manifesto
- **manifesto_id**: the id for the manifesto to which the segment belongs
- **party**: the political party that authored the manifesto
- **date**: the election date (year and month)
- **title**: the title of the manifesto

Columns are separated by commas (,). Strings that contain commas are delimited
by double quotes ("). Both files are encoded using UTF-8.
