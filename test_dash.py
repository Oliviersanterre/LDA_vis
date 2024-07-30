import dash
from dash import dcc, html
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel

# Sample documents
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey"
]

# Preprocessing the documents
def preprocess(text):
    return [word for word in text.lower().split() if word.isalpha()]

processed_docs = [preprocess(doc) for doc in documents]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Train the LDA model
num_topics = 2
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100, update_every=1, chunksize=10, passes=10, alpha='auto', per_word_topics=True)

# Prepare pyLDAvis visualization
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
html_string = pyLDAvis.prepared_data_to_html(vis_data)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Layout of the app
app.layout = html.Div([
    html.H1("LDA Model Visualization with pyLDAvis"),
    html.Iframe(srcDoc=html_string, style={"width": "100%", "height": "800px", "border": "none"})
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)