# CS172_Assignment2

Team members: Abraham Park, Masashi Yamaguchi

Programming Language: Python3

## Description
This program takes a list of queries and returns top 10 relevant documents for each query by applying cosine similarity.

## Dependencies
Install the dependencies:

```
pip3 install -r requirements.txt
```

## Usage
To run the program, run the following commands

```
python3 VSM.py --query query_list.txt --output output.txt
```

`--query` Reads the provided query file

`--output` Writes to the output text file provided

## Notes
Applied stemming and removed stopping words from query_list and documents.

Used binary weighting to evaluate cosine similarity
