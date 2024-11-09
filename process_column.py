
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

class TextProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
           # Tokenizer to split by space, period, underscore, and dash
        self.tokenizer = RegexpTokenizer(r'[^ \._\-]+')

    def tokenize(self, text):
        """Tokenizes the input text into words."""
       
        return self.tokenizer.tokenize(text)

    def case_fold(self, tokens):
        """Converts all tokens to lowercase."""
        return [token.lower() for token in tokens]

    def stem(self, tokens):
        """Applies stemming to the tokens."""
        return [self.stemmer.stem(token) for token in tokens]
# operate on a list of items
# and normalize each item and return the modified list
    
    def process(self, list_):
        """Combines tokenization, case folding, and stemming."""
        processed_res=[]
        # Tokenization
        for item in list_:
            temp=self.processString(item)
            merged_string = ' '.join(temp)
            processed_res.append(merged_string)

        return processed_res
    
    
    def processString(self, text):
        """Combines tokenization, case folding, and stemming."""
        # Tokenization
        tokens = self.tokenize(text)
            # Case Folding
        tokens = self.case_fold(tokens)
            # Stemming
        stemmed_tokens = self.stem(tokens)
        return stemmed_tokens
    
    
    def processColumns(self, columns):
         return [self.process(column) for column in columns]
        
        
    def columnsToBagOfTokens(self, columns):
        result=[]
        for col in columns:
            tokens=set()
            for xgrams in col:
                tokens.update(set(xgrams.split()))
            result.append(tokens)     
        return result
# Example usage
if __name__ == "__main__":
    text = "Tokenization is# - -the , process of breaking, down :text into smaller Units."
    processor = TextProcessor()
    processed_text = processor.processString(text)
    print(processed_text)