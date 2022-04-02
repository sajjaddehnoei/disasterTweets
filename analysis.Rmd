---
title: "Untitled"
author: "Sajjad"
date: "3/21/2022"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r SetOutputDirectory}
# This is necessary for knit_child to work properly when using "Run" in RStudio
knitr::opts_knit$set(output.dir = ".")
options(knitr.duplicate.label = 'allow')
```

```{r}
output <- knitr::knit_child("dataCleaning.Rmd")
```

`r output`

```{r libraries}
library(tidyverse)
library(stringr)
library(lubridate)
library(tidytext)
library(tm)
library(wordcloud)
library(wordcloud2)
library(irlba)
library(gridExtra)
library(keras)
library(tensorflow)
library(fpp2)
```

```{r readData}
train <- read_csv("train.csv")
test <- read_csv("test.csv")
```

# First, I'm Combining the two datasets to perform some cleaning. 
* note: technically, you're not supposed to do any processing before splitting  data.
```{r rBind}
x <- rbind(train%>%select(-target), test)
```

```{r cleaning__1}
x <- x %>%
  mutate(keyword = str_replace_all(keyword, "NA", "")) %>%
  mutate(keyword = str_replace_all(keyword, "%20", " ")) %>%
  mutate(keyword = str_to_lower(keyword)) %>%
  mutate(text = str_to_lower(text)) %>%
  mutate(text = gsub("http[^[:space:]]*", "", text)) %>%
  mutate(youTube = if_else(str_detect(text, "youtube"), 1, 0)) %>%
  mutate(mentions = if_else(str_detect(text, "@"), 1, 0)) %>%
  mutate(text = stemDocument(text, language = "english")) %>%
  mutate(text = str_trim(stripWhitespace(text)))
```

```{r cleaning__2}
# if a keyword is not mentioned in the text, add it to the text.
x <- x %>%
  mutate(text = if_else(!is.na(keyword) & !str_detect(text, keyword), 
                        paste(text, keyword, sep = " "), 
                        text))
```

```{r countwords}
x <- x %>%
  mutate(numStrings = str_count(text), 
         numWords = str_count(text, '\\w+'))
```

```{r removePunctuations}
x <- x %>%
  mutate(text = removePunctuation(text), 
         text = str_trim(stripWhitespace(text)))
```

```{r tokenize}
x <- x %>% unnest_tokens(word, text, strip_punct = T)
```

```{r abbreviations}
x <- x %>%
  left_join(abbreviations, by=c("word" = "abb")) 
  
x <- x %>%  
  mutate(word = if_else(!is.na(allWords), allWords, word)) %>%
  unnest_tokens(word, word)
```

```{r contractions}
x <- x %>%
  left_join(contractions, by=c("word" = "contraction")) 
  
x <- x %>%  
  mutate(word = if_else(!is.na(meaning), meaning, word)) %>%
  unnest_tokens(word, word)
```

```{r removeStopWords}
x <- x %>%
  anti_join(stop_words, by=c("word" = "word"))
```

```{r cleaning__3}
x <- x %>%
  group_by(id) %>%
  mutate(sentence = paste0(word, collapse =" ")) %>%
  mutate(keyword = if_else(word %in% unique(x$keyword) & is.na(keyword), 
                           word, keyword))
```

```{r cleaning__4}
# add sentiment as a word to the end of the sentence
x <- x %>%
  left_join(get_sentiments(), by= c("word" = "word")) %>%
  mutate(sentence = if_else(!is.na(sentiment), 
                            paste(sentence, sentiment, " "), sentence))
```

```{r cleaning__5}
df <- x %>% group_by(word) %>% summarise(n=n()) %>% arrange(desc(n))

# insight from data:
#   remove digits
#   remove special characters
#   remove < 10 and > 300
```

```{r cleaning__5}
x <- x %>%
  filter(!(str_detect(word, "[:digit:]")) & str_detect(word, "[:alpha:]")) %>%
  mutate(word = str_replace_all(word, "û", " ")) %>%
  mutate(word = str_replace_all(word, "ª", " ")) %>%
  mutate(word = str_replace_all(word, "ï", " ")) %>%
  mutate(word = str_replace_all(word, "ó", " ")) %>%
  mutate(word = str_replace_all(word, "å", " ")) %>%
  mutate(word = str_replace_all(word, "ã", " ")) %>%
  mutate(word = str_replace_all(word, "à", " ")) %>%
  mutate(word = str_replace_all(word, "ê", " ")) %>%
  mutate(word = str_replace_all(word, "â", " ")) %>%
  mutate(word = str_replace_all(word, "ò", " ")) %>%
  mutate(word = str_replace_all(word, "á", " ")) %>%
  mutate(word = str_trim(stripWhitespace(word))) %>%
     
  filter(word != "" & word != " " & word != "  " & 
         word != "i am" & word != "we are" & word != "they are",
         word != "he is" & word != "she is" & word != "we are not") 
  # %>%
  # anti_join(impWords, by = "word")
```


```{r trainSwitch}
xTrain <- x %>%
  filter(!duplicated(id)) %>%
  select(id, youTube, mentions, numStrings, numWords, sentence) %>%
  right_join(train %>% select(id, keyword, location, target), by = "id") %>%
  mutate(location = as.factor(location), keyword = as.factor(keyword)) %>%
  rename(text = sentence) %>%
  filter(!is.na(text))
```

```{r testSwitch}
xTest <- x %>%
  filter(!duplicated(id)) %>%
  select(id, youTube, mentions, numStrings, numWords, sentence) %>%
  right_join(test %>% select(id, keyword, location), by = "id") %>%
  mutate(location = as.factor(location), keyword = as.factor(keyword)) %>%
  rename(text = sentence) %>%
  filter(!is.na(text))
```

# Building the N Netword

```{r textVectorizer}
num_words <- nrow(x %>% filter(!duplicated(word))) + 1
max_length <-  150
# 140 is the maximum length for a tweet when the data was collected
text_vectorization <- 
  layer_text_vectorization(max_tokens = num_words, 
                           output_sequence_length = max_length)

text_vectorization %>% adapt(xTrain$text)

get_vocabulary(text_vectorization)
text_vectorization(matrix(xTrain$text[1], ncol = 1))
```

```{r buildModel}
input <- layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(nrow(x %>% filter(!duplicated(word))) + 1, 
                  output_dim = 256) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 256, activation = "relu", 
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)
```

```{r compileModel}
model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)
```


```{r trainModel}
history <- model %>% 
  keras::fit(
  xTrain$text,
  xTrain$target,
  epochs = 5,
  batch_size = 512,
  validation_split = 0.2,
  verbose=2)
```

# Logistic Regression
* To see the possible association between number of strings, number of words, mentions, and youTube links.

```{r buildModel2, eval = F}
lrModel <- glm(target ~ youTube + mentions + numStrings + numWords,
               data = xTrain, family="binomial")

summary(lrModel)
```

We can see from the Logistic Regression model that the following variables have a statistically significant effect on the outcome variable (p = 0.05):
* youTube
* mentions
* numStrings
* mentions

```{r lrFit}
lrModel <- glm(target ~  youTube + mentions + numStrings + num_words,
               data = xTrain, family="binomial")

summary(lrModel)
```

```{r lrEvaluate}
lrPredictions <- predict(lrModel, xTrain %>% 
                           select(youTube, mentions, numStrings, numWords), 
                     type = "response")

lrPredictions <- as.data.frame(lrPredictions) %>%
  rename(predictedProb = lrPredictions) %>%
  mutate(predictedValue=if_else(predictedProb > 0.500, 1, 0))

metric_binary_accuracy(xTrain$target, lrPredictions$predictedValue)
```

Logistic Regression has an accuracy of 0.62 just by looking at the following like:
* whether a tweet has a youTube link
* whether a tweet has a mention 
* number of strings in a tweet
* number of words in a tweet

# Predict the test set (Logistic Regression)
```{r lrPredict}
lrPredictions <- predict(lrModel, xTest %>% 
                           select(youTube, mentions, numStrings, numWords), 
                     type = "response")

lrPredictions <- as.data.frame(lrPredictions) %>%
  rename(predictedProb = lrPredictions) %>%
  mutate(predictedValue=if_else(predictedProb > 0.500, 1, 0))
```


# Predict the test set (Neural Network)
```{r predictTestNN}
predictions <- model %>% predict(xTest$text)

submission <- xTest %>%
  select(id) %>%
  cbind(predictions[,1])

names(submission) = c("id", "probNN")

submission <- submission %>%
  cbind(lrPredictions %>% select(probLR = predictedProb)) %>%
  rbind(tibble(id = 43, probNN = 0, probLR = 0)) %>%
#  rbind(tibble(id = 5810, probNN = 0, probLR = 0)) %>%
  mutate(target = 
           case_when(
             probNN > 0.5 & probLR > 0.5 ~ probNN,
             probNN < 0.5 & probLR < 0.5 ~ probNN,
             probNN > 0.5 & probLR < 0.5 ~ (2*probNN + probLR)/2,
             probNN < 0.5 & probLR > 0.5 ~ (2*probNN + probLR)/2)) %>%
  mutate(target = if_else(target >= 0.5 , 1, 0))

# submission <- submission %>%
# #  cbind(lrPredictions %>% select(probLR = predictedProb)) %>%
#   rbind(tibble(id = 43, probNN = 0)) %>%
# #  rbind(tibble(id = 5810, probNN = 0)) %>%
#   mutate(target = if_else(probNN >= .5, 1, 0))
```

# Export submission
```{r saveResults}
write_csv(submission%>%select(id, target), "submissionSajjad.csv")
```


* Note: to find an appropriate model size, it’s best to start with relatively few layers and parameters, then begin increasing the size of the layers or adding new layers until you see diminishing returns on the validation loss. Let’s try this on our movie review classification network.
* Here's some useful information on how to choose optimal number of hidden layers and nodes:
https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

