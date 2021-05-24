# CS172_Assignment2

Team members: Abraham Park, Masashi Yamaguchi
Programming Language: Python

## Description
This program takes the list of queries and returns top 10 relevant documents for each query by applying cosine similarity.

## Usage
To run the program, run the following commands
```
python VSM.py --query query_list.txt --output output.txt
```

`--query` Searches for specific query list
`--output` Searches for specific output file

## Dependencies


## Memo
Applied stemming and removed stopping words from query_list and documents.
Used binary weigting to evaluate cosine similarity
