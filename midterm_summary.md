## My Midterm Project Report

### Project Purpose:
This repo analyzes how S&P500 companies returns performed when and after their 10k report was released along with performing sentiment analysis to check whether or not these companies had positive or negative sentiment in their report and whether or not they would have higher or lower returns based off of this score.

### Summary Section
Summarize your question, what you did, and your findings. You can model this on the abstracts in the literature folder.

The main question for this project was "Do 10-K filings contain value-relevant information in the sentiment of the text?" Another relevant question was "Is the positive or negative sentiment in a 10K associated with better/worse stock returns?" In this project, I looked to answer these questions by seeing how companies in the S&P500 returns performed when and after their 10k report was released along with performing sentiment analysis to check whether or not these companies had positive or negative sentiment in their report and whether or not they would have higher or lower returns based off of this score. This took a lot of thinking before doing to say the least as well as doing some pseudocode in order to figure out what exactly I needed to write to get the solution I wanted. To begin downloading data in the ```get_text_files_PLAYGROUND.ipynb``` took some time and only downloaded 498 different firms and their corresponding 10ks. I will dive into why I think this happened below in the data section. From there, I pulled the CIK and Accession Numbers from the folder names so I could then pull CRSP stock data from 2022 only and manipulate it within the ```build_sample.ipynb``` file. Using many loops and merges I was able to create a dataframe called ```stocks_all``` that contained all of the stock return data I needed from the day of filing up(t) up until t+10 days following the 10k filing date. This would allow me to calculate the cumulative returns for the two seperate periods day t to day t+2 and day t+3 to day t+10. Then, I moved on to the sentiment section of the project calculating whether or not each of the 498 companies downloaded 10k had either a positive or negative sentiment through both an ML model and an LM model that contained a dictionary of positive and/or negative words. These scores were calculated by simply looping through each CIK and then opening and reading their corresponding HTML file for their 10k and checking how many positive or negative words based on the model being tested as well as the total number of words. I then calculated the sentiment score by simply dividing the two numbers either positive or negative by total words to get a sentiment score for that specific company. Next, I chose three different contextual sentiment topics I thought may be relevant in 10ks based on my prior knowledge. I chose strategic moves, leadership and governance, and industry trends and I'll explain more of why in the data section below, but on a broad scale I performed a contextual sentiment analysis on these topics using the ```NEAR_finder``` function testing to see if words surrounding these topics were found around positive or negative words and generating a sentiment score based on that. This would potentially show me how a company's stock price is reflected by the way they spoke about these topics in their 10k in a positive or negative way. This could potentially be interesting to see if there was any correlation and if one could make investment decisions based on these scores and returns. Finally, I merged all of these dataframes together into one final dataframe called ```final_stocks_sentiment``` that I could then test and see if there was any correlation between the sentiment scores I calculated and stock return data from the day of return as well as cumulative data following that date. This dataframe is saved in a file called "analysis_sample.csv" within the output folder. I will load that in below in order to create some charts and plots.

### Data Section
**What's in the sample?**

In my sample there are
- CIKS
- Filing Dates
- Tickers
- Returns on the Filing Date
- Cumulative returns from day t to day t+2 and day t+3 to day t+10
- Sentiment Scores for both ML and LM models based on the amount of positive or negative words in each 10K
- Contextual Sentiment Scores on strategic moves, leadership and governance, and industry trends
  - These scores were based on how often the words on each topic were surrounded by positive or negative words in each 10k

**How are the return variables built and modified?**

To begin building the return variables, I first merged the CIKs, Accession Numbers, and Filing Dates with the corresponding Ticker. This step was important because the SEC filing data was stored using CIKs in the dataframe ```df_cik_acc```, while the stock return data used Tickers. So I had to reference the file ```inputs/sp500_2022.csv``` and then do a left merge to ensure all firms in my ```df_cik_acc``` were retained even if some did not have matching stock return data. This method allowed me to keep the full list of companies from the original SEC dataset that I pulled while still adding all available stock tickers so I could then calculate return data. Once these were merged I began searching for the returns. Once I merged the dataset, I created a copy of ```merged_tickers``` so that any changes wouldn't directly change the original DataFrame as I had gotten a copywithwarning message. I checked the data types of both ```merged_tickers``` and ```stock_data``` to confirm that key columns like ```'Filing Date'``` and ```'Ticker'``` were properly formatted. Since the ```'Filing Date'``` column in ```stocks_all``` was initially a string, I converted it to a datetime format to match ```stock_data``` as I had recieved an error when initially trying to merge. I then merged ```stocks_all``` with ```stock_data``` on ```Ticker``` and ```Filing Date``` and renamed columns to make sure the merge worked. I then used a left join to keep all records from ```stocks_all``` even if some firms didn't have corresponding stock data. This pulled the stock returns from the date of filing. Then, I moved on to finding cumulative returns for both periods. Initially I started by simply creating all of the columns for returns t+1 - 10 in my dataframe and giving them a value of None as a placeholder. Next, I used a loop to iterate through each row in the dataset, identifying the filing date's position within each company's stock data. Once I found the filing date, I iterated forward to get returns for the next ten valid business days. To explain this part using my code, after iterating through each row in the dataset, I checked if the ```filing_index``` was not empty using ```if not filing_index.empty:``` to ensure that a valid filing date was found within the filtered stock data. If there was a match I set ```filing_index = filing_index[0]``` to extract the filing date's exact position for that stock and set ```next_bd_index``` as the following index ```(filing_index + 1)```. This would check for every day, but I still had to make sure weekends and holidays were accounted for. To ensure that the days were not weekends or holidays I added a while loop that skipped over non trading days. This loop checks if the current return value is NaN using ```pd.isna()``` and continues advancing the index until it finds the next valid trading day returns. I assigned the next valid return for each day to the proper column in ```stocks_all``` using this line of code which pulls the return from that specific date after the filing date, ```stocks_all.at[index, f'ret_t+{i}'] = filter_stock.at[next_bd_index, 'ret']```. If I couldn't find any valid return data I had it set the value to None. Finally, to calculate the cumulative returns I created a new column for each and pulled only the columns returns that were needed for each as seen through ```stocks_all[['ret', 'ret_t+1', 'ret_t+2']]``` or ```stocks_all[[f'ret_t+{i}' for i in range(3, 11)]]```. Then, I used ```.apply()``` with a lambda function to combine the individual returns as growth factors to align with how an investment compounds over time and accurately get the cumulative returns. For example the formula for the first cumulative returns would be *(1+ret)×(1+ret_t+1)×(1+ret_t+2)−1* so I simply applied this function in my code through this ```lambda x: (x + 1).prod() - 1```. To finish I simply sorted my values by CIK just to keep them in the order I had been working with initially.

**How are the sentiment variables are built and modified?**

To begin building the sentiment variables, I first imported predefined word lists that identified both positive and negative language patterns. For the BHR sentiment measures, I loaded two separate text files ```'inputs/ML_negative_unigram.txt'``` for negative words and ```'inputs/ML_positive_unigram.txt'``` for positive words converting all words using ```.lower()``` to ensure case sensitivity. For the LM sentiment measures, I imported the file ```'inputs/LM_MasterDictionary_1993-2021.csv'``` which holds both predefined positive and negative words in one file. This had to be seperated out using the code ```LM_positive = df[df['Positive'] > 0]['Word'].tolist()``` and the converted to ```.lower()``` for case sensitivity as well. And filled in negative after to seperate those out. Once these dictionaries were all loaded in I began the process of calculating sentiment scores. I loaded in the ```10k_files.zip``` and converted the word dictionaries into a regex pattern for each different dictionary through ```re.compile('|'.join(...))```. This code joins the list of words into a single string separated by the | symbol which allows searching through each 10k file. Following this I iterated through the 10K filings stored in a zip folder. For each file, I extracted the CIK and used ```BeautifulSoup``` to parse through each 10ks HTML content. Through this code ```match_count = len('...'_regex.findall(html_text))``` ```word_count = len(html_text.split())``` ```sentiment_score = match_count / word_count``` I was able to find all the positive or negative words and then calculate a sentiment score by dividing the total number of matched words by the total word count in the document. Each calculated value was appended to ```sentiment_scores``` with the corresponding CIK. I sorted by CIK to once again keep them in the same order. 

For the contextual sentiments this process was similar, but also included ```NEAR_finder() function``` which checked to see whether words in a list that I had created appeared near either positive or negative sentiment terms. To ensure this process was done accurately given you must specify a ```max_words_between``` I converted the text to lowercase and replaced extra spaces using ```re.sub(r'\s+', ' ', ...)```. To calculate how many matches I found in each document I wrote this line of code ```match_count, _ = NEAR_finder('...', '....', html_text, max_words_between=5, greedy=False)```. I chose 5 words because I felt that any further away may have been contracdictory and a new sentence regarding a new topic may have even begun. 5 seemed like the proper amount give typically when one would describe any of these topics, if it were positive or negative it would likely come right around that word. The underscore after ```match_count``` is simply just a placeholder or throw away variable as I only needed to get the number of matches not what these matches were. I set ```greedy``` to False as this made sure that each match that was found was kept seperate and not combined into one big match which could have happened if it was set to True. Once again I calculated the sentiment scores through the same calculation taking the number of matches divided by the total number of words and then appended that to ```sentiment_scores``` with the corresponding CIK.

**These datapoints about the sentiment variables:**
- How many words are in the LM positive dictionary?
  - Number of words in LM Positive dictionary: 347
- How many words are in the LM negative dictionary?
  - Number of words in LM Negative dictionary: 2345
- How many words are in the ML positive dictionary?
  - Number of words in ML Positive dictionary: 75
- How many words are in the ML negative dictionary?
  - Number of words in ML Negative dictionary: 94

```python
import pandas as pd

file_path = "inputs/LM_MasterDictionary_1993-2021.csv"
df = pd.read_csv(file_path)
LM_positive = df[df['Positive'] > 0]['Word'].tolist()
LM_negative = df[df['Negative'] > 0]['Word'].tolist()

with open('inputs/ML_positive_unigram.txt', 'r') as file:
    BHR_positive = [line.strip().lower() for line in file]
with open('inputs/ML_negative_unigram.txt', 'r') as file:
    BHR_negative = [line.strip().lower() for line in file]

lm_positive_words = len(LM_positive)
lm_negative_words = len(LM_negative)
ml_positive_words = len(BHR_positive)
ml_negative_words = len(BHR_negative)

print(f'Number of words in LM Positive dictionary: {lm_positive_words}')
print(f'Number of words in LM Negative dictionary: {lm_negative_words}')
print(f'Number of words in ML Positive dictionary: {ml_positive_words}')
print(f'Number of words in ML Negative dictionary: {ml_negative_words}')
```

**A description of how you set up the near_regex function (partial = true or false, distance = what) and why you chose the values you did.**

As I explained above, I set ```greedy=False``` in the ```NEAR_finder()``` function to ensure the shortest possible match between keywords and sentiment terms so matches that were found wouldn't be potentially combined. I also set ```max_words_between=5``` to capture relevant context without including too much unrelated information or missing any meaningful matches.

**Why did you choose the three topics you did for the “contextual sentiment” measures?**

I chose strategic moves, leadership & governance, and industry trends because I thought they could be potential topics discussed in 10ks that could hav a substantial impact on the future returns. Strategic moves such as mergers, acquisitions, and divestitures often are a sign that could reflect growth, risk, or operational changes, so I felt that if they were being positively or negatively talked about in a 10k then a company's returns would be reflected in that way. I also choce leadership & governance as it focuses on executive changes, board decisions, corporate policies, and more which are all of extreme importance when assessing a company's stability and direction. Finally, I chose industry trends as I wanted to see some broader market shifts, technological advancements, and possible pressures that often shape how a company positions itself within an industry. I felt as though these topics would be relevant in 10ks and thought that they could all have potential impacts on a company's returns.

**Show and discuss summary stats of your final analysis sample**
```python
df = pd.read_csv("output/analysis_sample.csv")
df.describe()
```

|       | CIK        | ret       | c_ret_t_t+2 | c_ret_t+3_t+10 | Negative BHR Sentiment Score | Positive BHR Score | Negative LM Score | Positive LM Score | Negative Strategic Sentiment Score | Positive Strategic Sentiment Score | Negative Leadership & Governance Sentiment Score | Positive Leadership & Governance Sentiment Score | Negative Industry Trends Sentiment Score | Positive Industry Trends Sentiment Score |
|:------|:-----------|:----------|:------------|:---------------|:-----------------------------|:-------------------|:------------------|:------------------|:-----------------------------------|:----------------------------------|:-------------------------------------------------|:------------------------------------------------|:-----------------------------------------|:----------------------------------------|
| count | 4.980000e+02 | 489.000000 | 498.000000  | 498.000000     | 498.000000                   | 498.000000         | 498.000000         | 498.000000         | 498.000000                         | 498.000000                        | 498.000000                                     | 498.000000                                  | 498.000000                             | 498.000000                            |
| mean  | 7.851046e+05 | 0.000742   | 0.003299    | -0.008148      | 0.038893                     | 0.042032           | 0.037060           | 0.014876           | 0.000565                           | 0.000427                          | 0.000382                                       | 0.000378                                | 0.000703                               | 0.000571                              |
| std   | 5.501943e+05 | 0.034294   | 0.051782    | 0.063944       | 0.004342                     | 0.005716           | 0.005268           | 0.002125           | 0.000353                           | 0.000465                          | 0.000157                                       | 0.000158                                | 0.000292                               | 0.000279                              |
| min   | 1.800000e+03   | -0.242779  | -0.447499   | -0.288483      | 0.023462                     | 0.028700           | 0.023828           | 0.008855           | 0.000000                           | 0.000000                          | 0.000112                                       | 0.000049                                | 0.000091                               | 0.000105                              |
| 25%   | 9.727650e+04	  | -0.016493  | -0.025156   | -0.047847      | 0.036300                     | 0.038399           | 0.033551           | 0.013495           | 0.000312                           | 0.000196                          | 0.000275                                       | 0.000267                                | 0.000509                               | 0.000374                              |
| 50%   | 8.825095e+05 | -0.001638  | 0.000000    | -0.006860      | 0.038993                     | 0.041591           | 0.036638           | 0.014756           | 0.000497                           | 0.000304                          | 0.000350                                       | 0.000360                                | 0.000664                               | 0.000530                              |
| 75%   | 1.136007e+06| 0.015826   | 0.027997    | 0.028006       | 0.041513                     | 0.044913           | 0.039942           | 0.016076           | 0.000760                           | 0.000455                          | 0.000448                                       | 0.000448                                | 0.000845                               | 0.000707                              |
| max   | 1.868275e+06| 0.162141   | 0.229167    | 0.332299       | 0.057010                     | 0.079866           | 0.059642           | 0.026648           | 0.002822                           | 0.004107                          | 0.001129                                       | 0.001256                                | 0.002410                               | 0.002179                              |


- Return Volatility
  - There is quite wide range in ```ret``` values (from -24.3% to +16.2%) and ```c_ret_t_t+2``` values (from -44.8% to +22.9%).
  - This is interesting to see how big of price swings happened following the filing dates.
  - These maybe show some reactions to the filings.
- LM Sentiment Scores
  - The negative LM sentiment scores are dominanting that of the positive sentiment scores.
  - The Negative LM Score has a mean of 0.0371, over 2.5x higher than the Positive LM Score mean of 0.0149.
  - I think this is related to the number of words in each dictionary. As said above there are 347 words in the LM positive dictionary compared to 2345 in the negative dictionary.
- ML Sentiment Scores
  - The Positive BHR Score has a mean of 0.0420 which is slightly higher than the Negative BHR Score mean of 0.0389.
  - Maybe this is suggesting that firms are trying to frame their 10ks in a more positive tone which would make sense as a more positive 10k would likely lead to better returns.

**Do your “contextual sentiment” measures pass some basic smell tests?**

To begin, yes there is noticable variation in the measures with some values all the way at 0 and others noticeably higher than that. For example, the max for ```Positive Strategic Sentiment``` is 0.0041 and it is 0.0028 for ```Negative Strategic Sentiment```. This variation shows that the measures are not the same and are accurately showing differences in language across different firms 10ks. Although the contextual sentiment scores are generally lower this aligns with what I expected given the short list of words for each topic. I started with 10 for the first one and increased it to 15, but I didn't notice any noticeable difference, so I would likely need to include a list of well over 50-100 words to really begin raising those scores. Despite this given that there are still spikes and variation in my sentiment scores for each company, I think that they are accurately identifying any matches or patterns in the language of the 10ks. Also, all of my topics are not industry specific, so I wouldn't expect any industry to perform better than another in my analysis, so there is no concern of any bias in my contextual sentiment scores.

**Are there any caveats about the sample and/or data? If so, mention them and briefly discuss possible issues they raise with the analysis.**

As I mentioned already, one caveat about my data is that my contextual sentiment measures used a small set of keywords which likely led to the lower sentiment scores, however they still show fluctuation so I think they are accurate. That said, I think if you expanded the word lists this would likely increase the scores simply by matching more language on each of the topics.

### Results Section

<img src="images/dummy_thumbnail.jpg?raw=true"/>

Four discussion topics:

On (1), (2), and (3) below: Focus just on the first return variable (which will examine returns around the 10-K publication)

On (4) below: Focus on how the “ML sentiment” variables (positive and negative) are related to the different return measures.

1. Compare / contrast the relationship between the returns variable and the two “LM Sentiment” variables (positive and negative) with the relationship between the returns variable and the two “ML Sentiment” variables (positive and negative). Focus on the patterns of the signs of the relationships and the magnitudes.

2. If your comparison/contrast conflicts with Table 3 of the Garcia, Hu, and Rohrer paper (ML_JFE.pdf, in the repo), discuss and brainstorm possible reasons why you think the results may differ. If your patterns agree, discuss why you think they bothered to include so many more firms and years and additional controls in their study? (It was more work than we did on this midterm, so why do it to get to the same point?)

3. Discuss your 3 “contextual” sentiment measures. Do they have a relationship with returns that looks “different enough” from zero to investigate further? If so, make an economic argument for why sentiment in that context can be value relevant.

4. Is there a difference in the sign and magnitude? Speculate on why or why not.

**1.** 
To compare and contrast these relationships, the LM sentiment measures show a negative relationship with returns, with Negative LM at -0.03 and Positive LM at -0.05. In contrast, the ML sentiment measures display a more mixed relationship where Negative BHR is positively correlated at 0.05, while Positive BHR shows a negative correlation at -0.01. All of these correlations are generally pretty weak, but they are quite interesting to look at. In the LM sentiment measures Positive LM has a stronger negative correlation than Negative LM which is odd because you'd typically expect positive language to reflect positively on returns, but thats not the case here. This is also somewhat true in the ML sentiment measures as we see that negative lanuage has a positive correlation to returns which is not what one would expect. These unexpected results may suggest that people interpret certain types of language differently than traditional sentiment assumptions in these dictionaries or there may be potential errors or limitations in my sentiment scoring process that impacted the results.

**2.**
My compare and contrast from above somewhat contrasts with Table 3 of the Garcia, Hu, and Rohrer paper. My LM sentiment measures somewhat aligned with Table 3 as they showed a negative relationship with returns just not at the same magnitude, however, the ML sentiment measures show a more significant conflict. In Table 3, Positive ML Sentiment shows a positive relationship with returns at 0.11 while Negative ML Sentiment shows a negative relationship -0.05. This differs from my analysis where Negative BHR Sentiment or Negative ML Sentiment has a positive correlation of 0.05 while Positive BHR Sentiment shows a negative correlation of -0.01. This difference suggests there may be issues with my sentiment scoring process as they don't follow suit. Given my results only somewhat aligned with their Table 3, the Garcia, Hu, and Rohrer study's larger sample size, extended time frame, and additional controls likely improved the reliability of their results. By including more firms and accounting for firm size and industry effects they were able to better find the true impact that positive or negative sentiment has on returns.

**3.**
The contextual sentiment measures show very weak correlations with returns showing that they have limited immediate impact. They don't look "different enough" from zero to really investigate further to see if they truly are making an impact. It would be quite difficult to make any economic argument as to why sentiment on any of these topics would be relevant in making a decision when investing. That said, I think if I created a larger dictionary of words to look for this would potentially increase the sentiment scores and could potentially be seen as a reflection in returns. This would lead to increased correlation, however, given the current numbers it would be very hard to say that these topics have a significant impact on return data.

**4.**
The ML and LM sentiment variables show distinct patterns as the LM measures have stronger negative correlations, while ML measures specifically the Negative BHR Sentiment shows a surprising positive correlation. This difference could be due to simply how each dictionary classified different language. Given all correlations are close to zero maybe this suggests that these lists include more neutral or both positive and negative language than they had meant to include. I think it's important to note that as the return window expands the correlations generally become stronger. This delayed increase in correlation as cumulative returns are added could suggest that the impact of sentiment actually takes some time to materialize meaning that investors are reacting more gradually to language in 10K filings.

[Back to Home Page](https://jonahnathan0.github.io/)
