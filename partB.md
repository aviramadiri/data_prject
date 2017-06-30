# final project #
### Lihi Verchik - 308089333 , Aviram Adiri - 302991468
---

## Part B:

After we got the data from Facebook (using 'Rfacebook' package) we can start with the data classification.

First, we will load the relevnt packages and data:

```{r}
setwd("C:/Personal/dataWorkshop/proj/data_prject")

```
And will read the data from the file we created in part A:

```{r}

posts <- read.csv("posts_with_genders.csv",na.strings = "")

```

###part 1 - pre-precessing

Before we can start with the classification, we need to split the messages to features.
After several tries, we understood that the optimal results are with the following features:
- 
- 
- 
- 


```{r}


```

Now we will write the processed data to a file:

```{r}
write.csv(posts_as_features, "posts_as_features.csv")
```

###part 2 - practice of the model:


```{r}


```

###part 3 - verify the model using the test data:


```{r}


```

###Conclusions:

