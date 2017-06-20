# final project #
### Lihi Verchik - 308089333 , Aviram Adiri - 302991468
---

## Part A:

First, we will load the relevnt packages and data:

```{r}
# install.packages('Rfacebook')
library('Rfacebook')
```

Note: the token for Facebook API is temporary, so the next code is probably not abailable anymore.

We used the method "getGroup" which returns posts of a specific group.
We looked for a big group with lots of posts every day, and we found "EDC Las Vegas 2017". The group-id of this group is 812215895553035.
after getting the posts, we removed the empty messages.

```{r}
fb_token = "EAACEdEose0cBAKXjXyim2PAhBmQ96L19SeI1oduBJPsdYFhq9kZCKog4vhvecMJPO2ZAlB4kUhB1rflfA1zjwmw03cFoTPlZC9SURN1zG1E6ggTi0lx24iRqyJzxsZC6n93CyZBYeAZBpCFpmQbefq49CsiwZC8jiOVM1UZCaGXBbvae5qkEm9ZA8ZCwQCUZCix7aMZD"

posts <- getGroup(group_id=812215895553035, token=fb_token, n = 1200, since =  '2017/01/01' )
posts = posts[rowSums(is.na(posts['message'])) == 0, ]

```

Now, after getting list of posts, we need to find the gender of each user.
First, we will get a vector of the writer's id of each message.
second, we will get the data of each user, using the method "getUsers".
Then, we will get the gender of each user.

```{r}
ids <- as.vector(posts['from_id'])

for (id in ids){
  users = getUsers(id, token=fb_token, private_info = TRUE)
}

genders <- as.vector(users['gender'])

```

Now we have the relevant data, and we just need to combine it.
We will add the gender column to the posts object, and then will get just the gender and the message columns:

```{r}
posts["gender"] <- genders

posts_with_gender = data.frame(posts$gender, posts$message)

```
We will write it to a file:

```{r}
write.csv(posts_with_gender, "posts_with_genders.csv")
```
And thats is! now we have the relevant data for the project, written on a file called "posts_with_genders.csv".

