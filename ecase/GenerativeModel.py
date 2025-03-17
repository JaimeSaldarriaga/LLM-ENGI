import openai



class GenerativeModelOpenAi:
  def __init__(self, api_key):
    self.client = openai.OpenAI(api_key=api_key)
  
  def create_prompt(self, date, combined_news, sentiment):
    prompt = (
      "You are a financial expert and Bitcoin investment analyst. Your goal is to provide actionable insights for investors.\n\n"
      f"Based on the following Bitcoin news from {date}:\n\n{combined_news}\n\n"
      f"Another language model has estimated that the overall sentiment of these news articles is {sentiment}.\n\n"
      "Analyze the sentiment and content of the news. Identify key topics, trends, and potential market impact factors.\n"
      "Provide expert-level insights on how these news articles may influence Bitcoin's price, considering both short-term and long-term investment implications.\n"
      "Finally, estimate the probability (as a percentage) that Bitcoin's price will rise or fall in the coming days, "
      "justifying your response based on the previous analysis.\n"
      "Conclude with a well-reasoned investment recommendation: Should an investor buy, sell, or hold Bitcoin given this news?"
  )
    return  prompt


  def create_prediction(self, date, df_merged): 
    df_filtered = df_merged[df_merged['date'] == date]
    try:
      combined_news = ' '.join(df_filtered['summary'])
      setiment_given_llm = df_filtered['sentiment'].mode()[0]
      prompt = self.create_prompt(date, combined_news, setiment_given_llm)
    except:
      return 'Not predicted'
      
      # Hacer una solicitud a la API
    response = self.client.chat.completions.create(
      model="gpt-4",
      messages=[{"role": "system", "content": "You are a financial expert specializing in Bitcoin and cryptocurrency investments."},
                {"role": "user", "content": prompt}])
    return response.choices[0].message.content
    

    


