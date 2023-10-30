from gensim.models import Word2Vec

# Example sentences related to e-commerce
sentences = [
    ["Customer", "service", "is", "key", "to", "success", "in", "e-commerce"],
    ["Product", "description", "should", "be", "detailed", "and", "accurate"],
    ["Delivery", "time", "is", "crucial", "for", "online", "shopping", "experience"],
    ["Customers", "often", "read", "product", "reviews", "before", "buying"],
    ["Payment", "options", "should", "be", "safe", "and", "convenient"],
    ["Recommended", "products", "are", "based", "on", "customer", "history"],
    ["E-commerce", "companies", "invest", "heavily", "in", "digital", "marketing"],
    [
        "There",
        "is",
        "a",
        "growing",
        "trend",
        "of",
        "buying",
        "fresh",
        "groceries",
        "online",
    ],
    ["Return", "and", "refund", "policy", "affects", "customer", "loyalty"],
    ["User", "interface", "of", "the", "website", "should", "be", "user-friendly"],
]

# Create Word2Vec model
model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, workers=4)

# Get word embeddings
word_embeddings = model.wv

# Get word vector for a specific word
word_vector = word_embeddings["e-commerce"]

# Find most similar words to 'e-commerce'
similar_words = word_embeddings.most_similar("e-commerce")

print("Word Vector for 'e-commerce':")
print(word_vector)

print("\nMost Similar Words to 'e-commerce':")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.2f}")

"""
In this proof of concept business scenario, word embeddings could be used in several ways such as 
understanding the context around certain terms like 'e-commerce', detecting topics of customer reviews, 
or predicting purchase behaviour based on previous browsing behavior. The words most similar to 'e-commerce' 
could also help in SEO or creating targeted marketing campaigns.
"""
