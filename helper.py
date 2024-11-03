from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
import emoji_data_python
from collections import Counter
import emoji
import regex
from textblob import TextBlob

extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df



def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        # Use regex to find emoji characters
        emojis.extend(regex.findall(r'\p{Emoji}', message))

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df



def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def predict_chat_activity(daily_timeline):
    # Prepare data for ARIMA
    data = daily_timeline.set_index('only_date')['message']

    # Initialize ARIMA model
    model = ARIMA(data, order=(5,1,0))  # You can adjust the order of ARIMA model as needed

    # Fit the model
    model_fit = model.fit()

    # Make future predictions
    forecast = model_fit.forecast(steps=30)  # Predict for the next 30 days

    # Create a dataframe for the forecast
    future_dates = pd.date_range(start=daily_timeline['only_date'].iloc[-1], periods=31)[1:]  # Next 30 days
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast}, index=future_dates)

    return forecast_df

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


def sentiment_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    sentiments = []
    for message in df['message']:
        blob = TextBlob(message)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            sentiments.append('Positive')
        elif sentiment < 0:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')

    sentiment_df = pd.DataFrame(sentiments, columns=['Sentiment'])
    sentiment_df['Count'] = 1
    sentiment_df = sentiment_df.groupby('Sentiment').count().reset_index()
    return sentiment_df


def message_length_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    message_length_df = pd.DataFrame(df['message'].apply(len), columns=['Message Length'])
    return message_length_df


def conversation_flow_analysis(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    conversations = []
    for i in range(len(df) - 1):
        if df.iloc[i]['user'] != df.iloc[i + 1]['user']:
            conversations.append((df.iloc[i]['user'], df.iloc[i + 1]['user']))

    conversation_flow_df = pd.DataFrame(conversations, columns=['User 1', 'User 2'])
    conversation_flow_df['Count'] = 1
    conversation_flow_df = conversation_flow_df.groupby(['User 1', 'User 2']).count().reset_index()
    return conversation_flow_df
