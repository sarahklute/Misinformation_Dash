import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from nltk import ngrams, word_tokenize
from collections import Counter
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
import io
import base64


# CLEANING THE DATA
def remove_stopwords(text, stopwords):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(cleaned_words)

def word_count(text):
    words = text.split()
    return len(words)

def get_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def clean_data(dataframe, stopwords_file):
    # Load stopwords from stopwords.txt file
    with open(stopwords_file, 'r') as file:
        stopwords = file.read().splitlines()

    # Convert the 'title' column to lowercase
    dataframe['title'] = dataframe['title'].str.lower()

    # Remove punctuation from the 'title' column using regex
    dataframe['title'] = dataframe['title'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove stopwords from the 'title' column
    dataframe['title'] = dataframe['title'].apply(lambda x: remove_stopwords(x, stopwords))

    # Calculate word count for each title
    dataframe['word_count'] = dataframe['title'].apply(word_count)

    # Calculate sentiment score for each title
    dataframe['sentiment_score'] = dataframe['title'].apply(get_sentiment_score)

    # Drop the 'source_domain' column
    dataframe.drop('news_url', axis=1, inplace=True)

# DATAFRAME
df = pd.read_csv('fakenewsnet.csv')

clean_data(df, 'stopwords.txt')

# Separate data for true (0) and fake (1) articles
true_text = df[df['real'] == 0]
fake_text = df[df['real'] == 1]

# Define sentiment score ranges
sentiment_ranges = {
    'Positive (0.5-1)': [0.5, 1],
    'Neutral (0-0.5)': [0, 0.5],
    'Negative (-1-0)': [-1, 0]
}

# CACLUCLATIONS FOR VISULIZATIONS
# Calculate the percentage of fake articles per source domain for the top 10 domains
top_domains = df['source_domain'].value_counts().index[:10]
fake_counts = {}
for domain in top_domains:
    domain_data = df[df['source_domain'] == domain]
    fake_counts[domain] = len(domain_data[domain_data['real'] == 0]) / len(domain_data) * 100

# Sort the top domains based on the percentage of fake articles
sorted_fake_counts = dict(sorted(fake_counts.items(), key=lambda item: item[1], reverse=True))


# Dropdown options for selecting sources
dropdown_options = [{'label': source, 'value': source} for source in df['source_domain'].value_counts().head(10).index]

# color palette for bar chart
blue_palette = ['#001F3F', '#003366', '#004080', '#00509E', '#0066CC', '#0077CC', '#0088CC', '#0099CC', '#00AACC', '#00BBCC']


# WORD CLOUDS
# Concatenate all titles into a single string
all_titles = ' '.join(df['title'])
real_complete = ' '.join(df[df['real'] == 0]['title'])
fake_complete = ' '.join(df[df['real'] == 1]['title'])

# Create and display the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Article Titles')


# Create and display the word cloud for real articles
wordcloud_real = WordCloud(width=800, height=400, background_color='white').generate(real_complete)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Real Article Titles')

# Create and display the word cloud for fake articles
wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_complete)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Fake Article Titles')

# Save the word clouds as images using Pillow
wordcloud_all_pil = wordcloud.to_image()
wordcloud_real_pil = wordcloud_real.to_image()
wordcloud_fake_pil = wordcloud_fake.to_image()


# BAR CHART
# Calculate the percentage of fake articles per source domain for the top 10 domains
top_domains = df['source_domain'].value_counts().index[:10]
fake_counts = {}
for domain in top_domains:
    domain_data = df[df['source_domain'] == domain]
    fake_counts[domain] = len(domain_data[domain_data['real'] == 1]) / len(domain_data) * 100

# Sort the top domains based on the percentage of fake articles
sorted_fake_counts = dict(sorted(fake_counts.items(), key=lambda item: item[1], reverse=True))



# DASHBOARDING
# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Misinformation Analysis"),  # Title
    html.Div([
        html.P("Discover the Spread of Misinformation in News Articles!"),

        html.P("Journalists have historically been considered the gatekeepers and curators of news and information. "
               "In the age of print media, journalists were heralded as the curators and watchdogs of the world. "
               "As the world moved online, print media began to struggle. This struggle is only growing. The news "
               "industry is struggling to maintain a readership base and stay afloat. Recent studies have found that "
               "attention spans have shrunk in recent years, likely due to social media. Most people, especially the"
               " next generation, no longer read through long, articulate, well researched news articles. According "
               "to the Washington Post, only 41% of individuals read past the headline, a number which has likely "
               "grown since the article was published in 2014. As bright colors and inviting images fill the screens "
               "and fight for clicks, journalists are forced to join in on this struggle to continue delivering news. "
               "In some cases, journalists may be forced to sacrifice their ethical principles and resort to misleading "
               "headlines and lazy reporting. "),



        # WORD COUNT VS MISINFORMATION
        html.H1("Word Count vs. Misinformation"),
        html.Div([
                        html.P(
                            "Most of the article title word lengths are concentrated around 10 words, regardless of "
                            "whether they contain true or misleading information. Interestingly, the articles with "
                            "misleading information are concentrated around a neutral sentiment while the article "
                            "titles with true information have a wider spread of sentiment."
                        ),
                    ]),
        dcc.Slider(
            id='word-count-slider',
            min=0,
            max=df['title'].apply(lambda x: len(x.split())).max(),
            step=1,
            value=0,
            marks={i: str(i) for i in range(0, df['title'].apply(lambda x: len(x.split())).max() + 1, 10)},
            tooltip={'always_visible': False}
        ),
        html.Div(id='slider-output-container'),
        dcc.Graph(id='word-count-graph'),



        html.Div([
            dcc.Graph(id='scatter-plot'),
        ]),

        html.Div([
            html.Button('Real', id='real-button', n_clicks=0, style={'fontSize': 18}),
            html.Button('Fake', id='fake-button', n_clicks=0, style={'fontSize': 18}),
            html.Button('Total', id='total-button', n_clicks=0, style={'fontSize': 18}),
        ], style={'textAlign': 'center'}),  # Center the buttons

        # SENTIMENT SCORE V RETWEETS
        html.Div([
            html.H1("Sentiment Score vs. Retweets"),
            html.Div([
                html.P(
                    "To better understand the rationale behind sentiment of news article titles, "
                    "examining retweets can help shed light on the word choices. Retweets can be indicative of "
                    "the popularity and number of views an article has. As news outlets struggle to stay afloat, "
                    "they will likely grasp on to any opportunity to gain readers attention and find new "
                    "readers for the article. "
                    "Comparing the true/false articles, for many news companies, articles that contain "
                    "misleading headlines lead to more retweets. This points to the idea that sensationalized, "
                    "misleading content does need to higher interaction, although unfortunate, journalists are "
                    "incentivized to create increasingly misleading headline to continue to maintain an audience."

                ),

            dcc.Dropdown(
                id='source-dropdown',
                options=[{'label': 'All Articles', 'value': 'All'}] + dropdown_options,
                value='All'
            ),

            dcc.Graph(
                id='sentiment-vs-retweets',
                figure={},  # You can provide the initial figure or leave it as an empty dictionary
                config={'scrollZoom': False},  # Disable scroll zoom for better interaction
                style={'height': '600px', 'width': '100%'}  # Adjust the height and width
            ),

                ]),
        ]),

        # BAR CHART
        html.H1("Top 10 Sites with Highest Concentration of Articles"),
        html.Div([
                html.P(
                    " Some sites more than others are responsible for misinformation. "
                    "These 10 sites are responsible for the highest concentration of misleading articles. "
                    "When examining which article titles are examples of misleading for sites, it is obvious "
                    "that tabloid, celebrity focused articles contain the most misinformation. The tabloid "
                    "sphere of news is most notorious for using unethical, misleading methods "
                    "to pull in viewers. "
                    ),
            ]),

        # dcc.Graph is included as a child of the layout
        dcc.Graph(
            id='top-sites-bar-chart',
            figure={
                'data': [
                    go.Bar(
                        x=list(sorted_fake_counts.keys()),
                        y=list(sorted_fake_counts.values()),
                        marker=dict(color=blue_palette),
                    )
                ],
                'layout': {'title': 'Percentage of Fake News Articles by Source Domain (Top 10)'}
            },
            config={'scrollZoom': False}  # Disable scroll zoom for better interaction
        ),

        html.Div(id='bar-chart-article-titles'),
        # conclusion
        html.P("The Conclusion: As generative AI becomes more prevalent and the media outlets continue to struggle, misleading "
               "information will become more and more of an issue. People are going to struggle finding reliable, "
               "well researched information and sifting through false or misleading information will become an "
               "important task. Studying the reasoning and methods behind this misinformation will allow future "
               "researchers and internet users to better understand fake news."),

    ])
])



# CALLBACKS FOR WORD COUNT GRAPH
@app.callback(
    Output('word-count-graph', 'figure'),
    Output('slider-output-container', 'children'),
    Input('word-count-slider', 'value')
)
def update_graph(selected_word_count):
    filtered_df = df[df['title'].apply(lambda x: len(x.split())) >= selected_word_count]
    misinformation_count = filtered_df[filtered_df['real'] == 1].shape[0]
    total_count = filtered_df.shape[0]

    # Calculate the percentage of articles with misinformation
    percentage_misinformation = (misinformation_count / total_count) * 100 if total_count > 0 else 0

    # Update slider output text
    slider_output = f"Articles with word count = {selected_word_count}: {total_count} articles. Misinformation: {misinformation_count}"

    # Create the bar chart
    fig = {
        'data': [
            {'x': ['Misinformation'], 'y': [percentage_misinformation], 'type': 'bar', 'name': 'Misinformation'}

        ],
        'layout': {
            'title': f'Word Count vs. Misinformation for Articles with Word Count = {selected_word_count}',
            'yaxis': {'title': 'Percentage of Articles', 'range': [0, 100]}
        }
    }

    return fig, slider_output

# CALLBACKS FOR REAL FAKE SENTIMENT SCATTER PLOT CLICKER GRAPH
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('real-button', 'n_clicks'),
    Input('fake-button', 'n_clicks'),
    Input('total-button', 'n_clicks')
)

def update_scatter_plot(real_clicks, fake_clicks, total_clicks):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'real-button':
        filtered_df = df[df['real'] == 1]
        title = 'Real Articles - Sentiment vs. Word Count'
        marker_color = 'green'
    elif button_id == 'fake-button':
        filtered_df = df[df['real'] == 0]
        title = 'Fake Articles - Sentiment vs. Word Count'
        marker_color = 'red'
    else:
        filtered_df = df
        title = 'All Articles - Sentiment vs. Word Count'
        marker_color = 'blue'  # Set the color to blue for the "Total" button

    # Set fixed ranges for the x and y axes
    xaxis_range = [df['word_count'].min(), df['word_count'].max()]
    yaxis_range = [df['sentiment_score'].min(), df['sentiment_score'].max()]

    # Adjust height and width of the scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df['word_count'],
        y=filtered_df['sentiment_score'],
        mode='markers',
        marker=dict(color=marker_color),
        text=filtered_df['title'],
        hoverinfo='text',
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title='Word Count', range=xaxis_range),
        yaxis=dict(title='Sentiment Score', range=yaxis_range),
        width=1500,  # Set the width of the scatter plot
        height=600,  # Set the height of the scatter plot
    )

    return fig

# CALLBACK FOR CLICKABLE BAR CHART OF TOP 10
@app.callback(
    Output('bar-chart-article-titles', 'children'),
    Input('top-sites-bar-chart', 'clickData')
)
def update_bar_chart_article_titles(clickData):
    if clickData is None or not clickData['points']:
        return ''

    # Extract the clicked label from the points array
    clicked_label = clickData['points'][0]['x']

    # Filter articles based on the clicked source domain
    articles_from_site = df[df['source_domain'] == clicked_label]['title'][:10]

    # Create a list of HTML list items for the article titles
    article_titles = html.Ul([html.Li(title) for title in articles_from_site])

    return article_titles


# CALLBACK FOR OVERALL SENTIMENT VS RETWEETS
@app.callback(
    Output('sentiment-vs-retweets', 'figure'),
    Input('source-dropdown', 'value')
)
def update_scatter_plot(selected_source):
    if selected_source == 'All':
        data = df
        title = 'Sentiment Score vs. Retweets (All Articles)'
    else:
        data = df[df['source_domain'] == selected_source]
        title = f'Sentiment Score vs. Retweets ({selected_source})'

    fig = px.scatter(data, x='sentiment_score', y='tweet_num', color='real',
                     color_discrete_map={0: 'blue', 1: 'red'},
                     labels={'sentiment_score': 'Sentiment Score', 'tweet_num': 'Retweets'},
                     title=title,
                     log_y=True)  # Set log scale for the y-axis

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)