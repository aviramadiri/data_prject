
# Final Project 

### Lihi Verchik - 308089333 , Aviram Adiri - 302991468

# Part A
In this part we will collect the data using Facebook API.
The token for Facebook API is temporary, so you may have some issues with executing the code.
Therefore, we didn't add this part to this file, since we want to prevent crashes.
We wrote the results from Part A in a file called "posts_with_genders.csv".

You can see the code here: [Part A](https://github.com/lolacoupons/data_prject/blob/master/partA.md)

# Part B
In this part we will create text classification model depends on the data we collect in part A

### First step: Reading and Organizing the data.

First, we will load the relevnt packages and data.
Then, we will read the data from the file we created in part A:



```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib
    


```python
import pandas as pd
import numpy as np
import matplotlib as plt
from bs4 import BeautifulSoup 
import re
import nltk
from nltk.corpus import stopwords
```


```python
df_posts = pd.read_csv("./posts_with_genders.csv",encoding = 'latin1')
```


```python
df_posts.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>Insomniac released a statement about the man's...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>How many people got scammed this weekend?????</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>Idk about yall, but to me the best set was All...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>Did you all know Kaskade performed at EDC yest...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>female</td>
      <td>https://www.facebook.com/jushonti.giberson/pos...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>male</td>
      <td>Bruh! &lt;ed&gt;&lt;U+00A0&gt;&lt;U+00BD&gt;&lt;ed&gt;&lt;U+00B8&gt;&lt;U+00B3&gt;...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>female</td>
      <td>I had a blast being Link on day 2. 70% called ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>male</td>
      <td>I feel bad for everyone still in line for the ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>male</td>
      <td>https://m.facebook.com/story.php?story_fbid=28...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>female</td>
      <td>One of the best time of my life !!! \r\rSee yo...</td>
    </tr>
  </tbody>
</table>
</div>



Then, we got the summary of numerical variables, and plot the histogram of ApplicantIncome.


```python
df_posts.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1092</td>
      <td>1092</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>1067</td>
    </tr>
    <tr>
      <th>top</th>
      <td>male</td>
      <td>Looking for a ride back to the airport that is...</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>656</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



We can see that we have lots of unknown chars in our text (emojis and so on).
The next method is used for cleaning the text.


```python
def post_to_words(row_post):
    # Function to convert a raw post to a string of words
    # The input is a single string (a raw post), and 
    # the output is a single string (a preprocessed post review)
    
    # 1. Remove tags
    row_post = re.sub('<.*?>', '', row_post)
    
    # 2. remove non-letters
    letters_only  = re.sub("[^a-zA-Z]", " ", row_post) 
    
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()     
    
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))   
    
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops] 
    
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 
```

Lets check our method:


```python
clean_post = post_to_words(df_posts["message"][0] )
print(clean_post)
```

    insomniac released statement man death
    

And now let's apply this method on the posts:


```python
clean_posts = df_posts['message'].apply(lambda post: post_to_words(post))
clean_posts.head(10)
```




    0               insomniac released statement man death
    1                      many people got scammed weekend
    2    idk yall best set allison wonderland b b diplo...
    3           know kaskade performed edc yesterday effff
    4    https www facebook com jushonti giberson posts...
    5                                   bruh hope hurt bad
    6                      blast link day called zelda tho
    7                feel bad everyone still line shuttles
    8    https facebook com story php story fbid id us edc
    9      one best time life see next year electronic sky
    Name: message, dtype: object



### bag of words - BOW

The Bag of Words model learns a vocabulary from all of the documents, then models each document by counting the number of times each word appears.


```python
print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 5000) 

# fit_transform() Convert a collection of text documents (reviews in our example) to a matrix of token counts.
# This implementation produces a sparse representation.
# The input to fit_transform should be a list of strings.
posts_data_features = vectorizer.fit_transform(clean_posts)
###train_data_features = vectorizer.fit_transform(train['review'])

# Numpy arrays are easy to work with, so convert the result to an 
# array

posts_data_features = posts_data_features.toarray()
```

    Creating the bag of words...
    
    


```python
vocab = vectorizer.get_feature_names()
print(vocab[0:50])
```

    ['able', 'absolute', 'absolutely', 'ac', 'acceptable', 'access', 'accident', 'accidentally', 'account', 'achey', 'acontext', 'across', 'action', 'actively', 'actually', 'add', 'adding', 'additional', 'ade', 'admiring', 'admission', 'admissions', 'advantage', 'adventure', 'af', 'affordable', 'afrojack', 'afternoon', 'afterwards', 'age', 'agencies', 'ago', 'agrees', 'aid', 'aint', 'air', 'airbrush', 'airport', 'airways', 'albuquerque', 'alcohol', 'alesso', 'alex', 'alexandria', 'alive', 'allison', 'alllove', 'allow', 'allowed', 'allready']
    

### Second step: Training the Model following the train data.
Now, after all the information is completed, we can start with the Training part. 
First, we will split the data to test and train:


```python
#split to train & test
train_posts = np.random.rand(len(df_posts)) < 0.75
train_message = posts_data_features[train_posts]
train_gender = df_posts.loc[train_posts,"gender"]

test_message = posts_data_features[~train_posts]
test_gender = df_posts.loc[~train_posts,"gender"]
```

Let's check several models:

### First model:  K-Neighbors model.


```python
test_gender.head(10)
```




    2       male
    3     female
    5       male
    11      male
    12      male
    14      male
    17    female
    20      male
    21    female
    28      male
    Name: gender, dtype: object




```python
from sklearn.neighbors import KNeighborsClassifier

KNeighbors = KNeighborsClassifier(n_neighbors=130) 

KNeighbors = KNeighbors.fit( train_message, train_gender )

score = KNeighbors.score(test_message,test_gender )
score
```




    0.59859154929577463



Result - in the best case we got ~ 0.65

### Second model: Gradient Boosting model.


```python
from sklearn.ensemble import GradientBoostingClassifier

GradientBoosting = GradientBoostingClassifier( n_estimators = 45 ) 

GradientBoosting = GradientBoosting.fit( train_message, train_gender )

score = GradientBoosting.score(test_message,test_gender )
score
```




    0.61052631578947369



Result - in the best case we got ~ 0.63

### Third model: Decision Tree model.


```python
from sklearn.tree import DecisionTreeClassifier

DecisionTree= DecisionTreeClassifier(random_state = 1) 

DecisionTree = DecisionTree.fit( train_message, train_gender )

score = DecisionTree.score(test_message,test_gender )
score
```




    0.54385964912280704



Result - in the best case we got ~ 0.62

### 4th model: Logistic Regression model.


```python
from sklearn.linear_model import LogisticRegression

LogisticRegression= LogisticRegression() 

LogisticRegression = LogisticRegression.fit( train_message, train_gender )

score = LogisticRegression.score(test_message,test_gender )
score
```




    0.58245614035087723



Result - in the best case we got ~ 0.56

### 5th model: Random Forest model.


```python
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier= RandomForestClassifier( n_estimators = 80 ) 

RandomForestClassifier = RandomForestClassifier.fit( train_message, train_gender )

score = RandomForestClassifier.score(test_message,test_gender )
score
```




    0.60915492957746475



Result - in the best case we got ~ 0.61

## Summarize
The best result that we got was with K-Neighbors model, with score of 0.65 .

We expected to better results, but we assume that the reason for that is our data- it may be too monotonous and not enugh diverse, eventhought we took it from a public group.


# Part C
In this part we will create text generation model depends on the data we collected in part A

### First step: Reading and Organizing the data.

First, we will load the relevnt packages and data.



```python
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
```

Now, we will split our data to two groups - males and females (we want to practice each group separetly). 


```python
male_posts = df_posts[df_posts.gender == 'male']
female_posts = df_posts[df_posts.gender == 'female']

```

Now we will clean the data (as we did in part B):


```python
clean_male_posts = male_posts['message'].apply(lambda post: post_to_words(post))
clean_male_posts.head(10)
```




    1                       many people got scammed weekend
    2     idk yall best set allison wonderland b b diplo...
    5                                    bruh hope hurt bad
    7                 feel bad everyone still line shuttles
    8     https facebook com story php story fbid id us edc
    10    hey guys took videos big group us double tripp...
    11    anyone driving phx sometime today tomorrow rid...
    12    credit card statements bank balances looking l...
    13           anyone flight delays wanna edc day airport
    14            else agrees merch edc year fucking sucked
    Name: message, dtype: object




```python
clean_female_posts = female_posts['message'].apply(lambda post: post_to_words(post))
clean_female_posts.head(10)
```




    0                insomniac released statement man death
    3            know kaskade performed edc yesterday effff
    4     https www facebook com jushonti giberson posts...
    6                       blast link day called zelda tho
    9       one best time life see next year electronic sky
    15    lost item edc unable make contact lost found d...
    17    many people sick head cold edc next year takin...
    18                      someone vegas help get weed wax
    21                                        ok watch lmao
    23    really sorry people good time stellar time din...
    Name: message, dtype: object



Now we will convert it to text:


```python
male_posts_as_text = clean_male_posts.str.cat(sep=".")
print(male_posts_as_text)
```

    many people got scammed weekend.idk yall best set allison wonderland b b diplo b b jauz also joyride killed shit doooope yalls favorite set.bruh hope hurt bad.feel bad everyone still line shuttles.https facebook com story php story fbid id us edc.hey guys took videos big group us double tripple even quad light shows please post videos would love see green lights gloves dakota outspoken balthrop branson outspoken balthrop jenn marie light show crew tag us videos thank u guys first year edc loved thanks changing lives edc family.anyone driving phx sometime today tomorrow ride somehow left yesterday swear knocked tf.credit card statements bank balances looking like edc.anyone flight delays wanna edc day airport.else agrees merch edc year fucking sucked.ok cried axwell ingrosso music really reintroduced scene played reload broke crying writing.water free water shuttle downtown.hey hey hey ride back airport extremely affordable charging station inside order lyft promo code edcvegas.dj slander.hey guys good morning apparently flight got cancelled till tomorrow morning one help please low funds expecting need somewhere stay day please thank give guys.edc doesnt time fun let go forget everything meet new friends trade kandis get fucked lets try make life fun edc edc lifestyle im vip travel club pays want im love job want help people travel help children people need im always looking dope people join miami next im leaving la thursday everyone going vacation summer say help get cheapest price expenses let know dont give fuck anyone says traveling experience one things makes feel alive everyone says looking purpose life maybe looking experience make feel alive physical level shy dream possibilities world huge whatever think pretty sure ask please igotnowfuckwhogotnext listen homies mix https soundcloud com geppettousa geppetto discofamhq resident mix guz add snap pocahotness.meet coolest chick dude whatever ever met edc n probably never gonna meet damn marshmello.samurai siriacha.hey hey hey ride back airport extremely affordable charging station inside order lyft promo code edcvegas.ready edc day.drake edc death edc thanks pasquale.friendly reminder folks make sure air completely dry inside camel pack including hose make sure take ends dry properly end moldy pack next time want use.anyone see.one thought many fireworks year compared last couple years wanted see.litty af.anyone luxor hotel.hey hey hey ride back airport extremely affordable charging station inside order lyft promo code edcvegas.looking ride back airport extremely affordable charging station inside order lyft promo code.main firework show seven lions.ok cried axwell ingrosso music really reintroduced scene played reload broke crying writing.marshmello lit.looking ride back airport extremely affordable charging station inside order lyft promo code.shout old guy kept roll killin markus schulz.https djmag com news edc las vegas sets ti sto armin van buuren major lazer watch.https facebook com story php story fbid id.anyone circus circus hotel resort tryna chill.thank onenonlyent staff attendees made one lifes monments significant getting us going bat us wonderful friendships.anyonw still vegas wanna hangout.honestly fuck insomniac shuttles im asking least partial refund shit fucked also used village stop sorry.porter w fireworks.looking ride back airport extremely affordable charging station inside order lyft promo code.need find two girls diplo tonight encore get guest list.lol good one.photographer still town want go shoot milky way painted rocks tonight also forgot tripod maybe could share.might still feeling high edc experience wrote impassioned plea today hope read.looking ride back airport extremely affordable charging station inside order lyft promo code.hey fam need help bus left like people stranded left earlier departure time get another one till tomorrow anyone place stay tonight smoke fat like point idc floor leave tomorrow morning help.showtek dope af.felt saw datsik day.disappointed going get married purchased vip tickets included personal locker got told us locker somehow booked went edc town chapel noticed chapel scheduled get married lights working top shuttles absolute nightmare poorly ran expected better insomniac needless say got married somewhere else.get memo glitter beards saw like dudes lmao.looking ride back airport extremely affordable charging station inside order lyft promo code edvvegas.edc finest lmao fuckgettingahotelilljustcrashhereonthistrashcan.ready hit studio.looking ride back airport extremely affordable charging station inside order lyft promo code.looking ride back airport extremely affordable charging station inside order lyft promo code.solve ur post edc blues episodes rave train sets ur fav.looking ride back airport extremely affordable charging station inside order lyft promo code.looking ride back airport extremely affordable charging station inside order lyft promo code.hope everyone safe trip back home.festival awesome way uber drop horrible long think moises crossing desert hard.free guest lists available edc family still town ratio group size free guest list text.selling male one female diplo ticket xs tonight.follow official edc babes snapchat rave girls parties please submit pictures chance win rave girl edc.follow official edc babes snapchat rave girls parties please submit pictures chance win rave girl edc.looking ride back airport extremely affordable charging station inside order lyft promo code.case nobody knew one black shirt said wall since back shirt add saw.fuck shuttle line joke went catch uber nobody told us signed edc marathon anyone know collect finish times thanks.anyone space hotel two monday night tuesday pay.always surround good vibes next time.one edc.anyone encore wynn pallazo venetian want hangout.fucked ravers show love help win dillon francis remix comp free merch win safe people much love plur.sure download lyft app store enter code edcvegas large discounts rides avoid crazy shuttle lines pleasure arriving fully charged phone.wow min waiting village luxor shuttle wow.last minute sunday band ga available strip meet well edc.day solo trip hopefully connect beautiful souls.waited less mn get shuttle perfect.sure download lyft app store enter code edcvegas large discounts rides avoid crazy shuttle lines pleasure arriving fully charged phone.anybody want meet trade kandi tonight.hello humans also human looking wristband tonight help.looking two edc tickets paying.going hour getting premier parking taken hrs get parking spot hollywood blvd fd situation nobody directing traffic parking lot free.anyone selling legit tickets.sure download lyft app store enter code edcvegas large discounts rides avoid crazy shuttle lines pleasure arriving fully charged phone.last call take group people passenger van leaving strip mins message.keeps popping feed passing along anyone needs.looking two sunday wristbands paying wristband.passenger van us going take interested message details.bull shit took get tonight get wristband tomorrow sale make offer let take pic license ever want coming back shit show.need wristbands tonight anyone selling thank.wristbands valley center ca wristbands sale leave hotel.rides edc today willing roll blunts ride ppl.free guest lists available tonight tomorrow day free entry ratio list hit.need ticket please help.free rider credit lyft https www lyft com invite theron first time riders.sure download lyft app store enter code edcvegas large discounts rides avoid crazy shuttle lines pleasure arriving fully charged phone.life gets rough lol.post snapchat name mine gremlin let see edclv videos pictures also outfit day.premier shuttle passes san francisco ca two premier shuttle passes sale bucks come get mirage.need sunday wristbands got.need two vip edc tickets box.kicked guy shin yesterday raging said sorry shook hand kept raging dude cool shit another apology friend.shuttle passes leaving festival ground.sunday pass vegas village shuttle meet delano suites.final day metro booming want.one ga wristband one ga wristband sunday linq shuttle pass message meet around linq.sure download lyft app store enter code edcvegas large discounts rides avoid crazy shuttle lines pleasure arriving fully charged phone.selling edc wrist band pm info.looking edc tickets.edc shuttle passes village ceaser palace two shuttle passes.anyone selling tickets today hmu please.village shuttle pass new york new york hotel casino las vegas las vegas village shuttle pass sale message interested ny ny right.last call party yall tree nd telsa left get em hit traffic.still needing tickets edc sunday.ga tickets green bay wi friend selling ga tickets tonight pm interested.two linq shuttle passes sale dm.need two sunday tickets unregistered cash come pm.looking edc tickets sunday.anybody selling ticket tonight need one wife renew vows church tonight thanks plur.sunday passes sale low need heeeelp please.sunday ga ticket sale hmu linq shuttle pass also hmu offers.hey anyone taken taxi uber long took need shuttles seem reliable.understand lot upset mad long wait times shuttles us vets warned st timers first started shuttles fam waited dam near almost pm first night waiting took cab uber event split difference sorry ruined experience edc future decide come back better go late early missed sat due slight mishap see tonight.sunday tickets sale.premier shuttle passes henderson nv selling two premier shuttle passes together.edc wristband las vegas nevada selling one edc wristband must pick.shuttles hmu.selling two shuttle passes luxor cheap.wristband degrees monte carlo sunday wristband shuttle pass prefer meet front monte carlo.goth raver chicks.sure download lyft app store enter code edcvegas large discounts rides avoid crazy shuttle lines pleasure arriving fully charged phone.vegas village shuttle passes sun city ca las vegas village shuttle passes anybody wants decided drive today need.best buffet las vegas need energy day.selling sunday vip unregistered band pickup monte carlo.sunday ticket unregistered pm interested.got sunday ticket lmk.two premier shuttle passes need hmu.looking.rave babe today go access great time pm interested.needing one unregistered wristband message.tickets cirque du soleil selling tonight contact good discount https www facebook com groups.day many great memories allready sad end near.got nothing going today gabberfest djs going hard hat lounge industrial new york near strip buys first drink gets show pm inside outside must.anyone tropicana wanna grab lunch together.selling one edc sunday ticket.leaving vegas la one hour providing rides gas money seats available.female.two premier shuttle passes sale craziness shuttles managed make back ease send message interested staying mirage.super towing llc cpcn.need wristband pick hour lmk soon.good morning fellow edc attenders first time edc las vegas seemed lost phone samsung galaxy black front white back honest someone try steal severe black screen issue freaking give needed outdated phone badly simply want memories able capture phone still fond memories even edc still someone happens see post found phone please turn lost found thank guys much taking time read love guys great energies enjoy tonight rest lives.two wristbands available one go big wrist needs girl small wrist unregistered.sunday wristbands valley center ca sunday wristbands sale meet hotel drive use venue walk together avoid traffic interested ready leave pm sharp pm interested.selling sunday edc pass let know intrested.anyone linq shuttle pass sale.selling day pass shuttle bus lmk interested meet vdara hotel.missed delta heavy kalliope last night fucked.looking rave babe tonight would like go access pm interested.heads guys linq shuttle drop back strip several undercover cops selling drugs someone getting arrested also side.even want ask village shuttle last night loool.looking one sunday ticket shuttle las vegas village.u get shit wrist band security tighten trying sell.free rider credit lyft new riders.group still extra unregistered ga vip tickets make offer.anyone know ubers get use route shuttle buses take leaving speedway like nellis route.search edc sunday wristband hit price cash hand.village shuttle passes best offer takes.making memories.one take uber lyft speedway much cost.edc needs serious work return shuttles walk mile get shuttle fast lane working better shuttles still need fine tuning.got shuttle pass linq sale.linq shuttle pass selling linq shuttle pass longer going sunday free offers must pick linq hotel.needing tickets sunday edc staying venetian.anyone selling edc sunday ticket.got vegas got cash hand willing meet buy ticket sunday buy sketch.looking buy edc wristbands tonight sat sun wristband looking purchase around max paying live lasvegas transportation meet within mins let know available wait dm vegas lasvegas electricdaisycarnival.aye got cash hand sunday ticket dude flaked vegas local meet.went last night go tonight think make third selling two sunday tickets near offer message interested edit lot traffic post hmu best offer otherwise list people want price.need ride edc tonight tomorrow morning edc.current situation shuttle fml.selling two sunday ticket.looking ticket sunday.anyone las vegas village shuttle yet ts bad yesterday want waste time walking would rather uber bad please let know.looking tickets today sat.looking edc ga tix cash vegas.vip bands unregistered vip bands ga unregistered band box pick next hour caesars palace.shuttle buses looking time everyone someone plz update.linq shuttles going.looking sat tickets sat sun tickets.https www facebook com events acontext b ref c action history null.anyone lv selling edc wristband tonight sunday.slowly surely getting see us dont shy come say hi.night let get ladies gentlemen.anyone need lift product pm.looking saturday ticket.looking couple sunday tickets inbox plsssss.steve aoki tonight hakkasan nightclub irie balling state champs jewel nightclub martin garrix omnia nightclub free entry guestlist guys girls hmu pm pm reservations jerry ngo promotions manager hakkasangroup instagram jerryngo viplasvegas hakkasan l omnia l jewel l wet republic ultra pool l.selling las vegas festival shuttle pass message interested.one village lot shuttle pass sale hmu.need wristbands remaining days helpppppppp.looking buy edc wristbands tonight sat sun looking purchase around know usually sell little plan going till later tonight sell wristbands still waiting buy still live lasvegas transportation meet let know available wait dm vegas lasvegas electricdaisycarnival.sunday tickets shuttle sunday tickets shuttle downtown pass obo.looking sunday tickets.looking shuttle passes pls msg.anyone vegas selling edc tickets sat sunday need meet anywhere.need two wristband sat got.looking sunday ga pass meet vegas anywhere hit.looking sat ticket.long take get last night drop.pm.whats pregame spot tryna blaze n grab drink.looking buy edc wristbands tonight saturday looking purchase around know usually sell little plan going till later tonight sell wristbands still waiting buy still live lasvegas transportation meet let know available wait dm vegas lasvegas electricdaisycarnival.anyone selling passes tonight got two people wanting.lyft uber expensive tonight tomorrow night get linq passes two.want buy around sunday edc tickets.brother another mother.one person closes gaps emptiness feeling unexplainable.need ticket tonight could return sunday willing pay text.looking nice saturday ticket return sunday text air right land.coming omnia axwell n ingrosso tonight free guest list comment dm.leaving tonight selling wristband linq shuttle pass.anybody know fuck goin wit las vegas village shuttle shit nightmare yesterday.fuck traffic situation last night.first line day edc.looking saturday sunday wristband.looking two saturday tickets.anyone vegas selling edc tickets sat sunday need meet anywhere vegas.villiage passes alexandria va got villiage passes sat sun pick linq.fuck ready day.looking saturday sunday pass.two village shuttle passes buy meet caesars palace.know long shot lost mgmshuttle pass last night chance anyone saw picked friend got sick trying help med tent probably dropped stuffs somewhere circus ground med tent get enjoy anything last night hopefully tonight better find shuttle pass get thank much return also sweet couples helped carrying friend last night guys heroes going way helping us.calvin harris tonight hakkasan nightclub gta jewel nightclub axwell ingrisso omnia nightclub limited entry guestlist guys girls hmu pm pm reservations jerry ngo promotions manager hakkasangroup instagram jerryngo viplasvegas hakkasan l omnia l jewel l wet republic ultra pool l.looking sat sunday pass.selling las vegas festival shuttle pass message comment interested.traffics bitch vegas real lanes lot idiots going stay.got day passes edc shuttle passes village single ticket shuttle pass takes obo.attention anyone going edc tonight would like come omnia nightclub see axwell ingrosso free comment dm.day vip pass meet cosmo.las vegas local looking purchase sunday tickets meet anytime anywhere cash via paypal pm selling.download lyft input code promos get cheap rides avoid crazy shuttle lines.need edc tickets tonight got deals hmu asap.sat excalibur hotel casino sat sun edc must meet excalibur hotel paypal pm serious inquiries please.looking saturday sunday ticket pick right staying excalibur.wristband saturday sunday hmu interested.sapphire pool day club text.ga saturday sunday anyone needs.parents selling wristbands tomorrow leave tomorrow afternoon hmu details bands unregistered.lit.best time leave edc pm.need saturday pass preferably anyone.anybody else saw guys getting lit lol.mgm shuttle selling sunday wristband mgm shuttle pass hit interested.test kit.basspod tits ln big ups fellow headbangers.selling two premier shuttle passes tonight tomorrow staying mirage message details.hi las vegas local looking edc passes tonight saturday.yall like.anyone selling sunday ticket willing let go please hmu vegas also hopefully makes easier.selling edc sunday ticket.need one ga ticket one mgm shuttle pass sat sun hit meet vegas.anyone want trade linq shuttle pass las vegas village lot tomorrow.tickets still couple brand new unregistered ga tickets box selling first come first served monte carlo.anyone going day willing sell shuttle pass sunday linq would ideal open shuttle.need ga day passes cash hand.hunting edc ticket still meet strip hotel ever.edc tickets shuttle passes st paul mn selling two edc tickets shuttle passes luxor tonight sat june sun june message details.couple ga wristbands boxes anyone needs one.hey guys whats app ive seeing one cam message thru bluetooth dont need signal use.saturday sunday vip edc ticket.male ticket tiesto pool party today wet republic mgm firm paid plus fees staying excalibur cant go anymore.village lot passes clermont fl village lot passes.band lucerne ca sale edc band day sister like edc want come back obo meet vegas tomorrow morning afternoon.anyone know another way get wristbands straw trick trying morning luck.one ticket sale.got tree guys pm.leaving vegas la monday morning around anyone needs ride home ask gas money good vibes.shower wristband lol.looking sunday bracelet.else rolling solo today.hands traffic jam.looking saturday sunday wrist band vegas right.https soundcloud com tableofcontents better left unsaid coverwritten elisha nims produced recorded tableofcontents.driving venue good time start heading strip ga parking read traffic awful thanks bunch.ps edc wristband.phiso b b ponicz absolutely efffin killed.calling shei ac selling tickets via paypal call box office get box office say ordered number tickets sold people filed fraud charge cancelled ticket orders get money back insomniac keep people payed getting written solid proof anyone else falls victim way reverse paypal transactions ease.looking saturday passes.fucking lit.selling two premier shuttle passes pick mirage.hrs get holy shit traffic nuts.use lyft code download app input code promos hugely discounted ride.hey throwing party place edc since go anyone interested dm deets bring party favors far strip.hey guys friend lost wristband anyone know people selling tickets door edc.friday wristbands pm.anybody day pass.anyone oregon headed edc ride bailed str hitchhiking lol makes one hell story south roseburg got spot happy trails.ticket tmr wanted contact asap u one.looking saturday sunday sunday ticket hmu.anyone going edc tn.fun everyone safe wish could make year rave guys post family photos us make.one braclet left register hit info.need promoter help booking spot weekend asap group connections.friday ticket san diego ca friday ticket obo deposit required insure get wristbands back otherwise use deposit get replacement pm interested.looking friday ticket hmu.plur fam anyone still need place call home weekend hmu.friend pick pass call.looking edc friday.michael alex officially landed.flying seattle vegas see soon.need shuttle passes got em.leaving mgm.camel packs empty fill ice.go get replacement wristband friend mine fucked know.anyone willing trade linq shuttle pass downtown shuttle pass text.phone plus unlocked good condition gb plus michael vick jersey trade two edc tickets hmu thoooo interested.stay safe drink plenty water.stay hydrated people stress enough.ride credit download lyft app using referral link https www lyft com john affordable ride town terms apply.party busses edc byob pick times pm pm back dm info.one edc ticket las vegas strip selling one edc ticket fv msg interested.day vip pass.high roller tickets las vegas nv high roller tickets.many asian girls lol feels like heaven.hey anyone oneandonly shuttle pass sale trying get hold days luck crew shuttle passes like one.linq shuttle san francisco ca friend selling linq shuttle pass obo.looking edc ticket meet vegas.whos looking needing edc tickets.village lot shuttle pass dm.looking edc passes festival ground shuttle pass.looking lv village shuttle pass.call still flamingo.edc hotel deals start july st book early get room price people paying dates change always rebook days prior check cancel always offer better rate meaning find cheaper rate send link beat rate group rates available hotels friendly www hotelsedclasvegas com sponsored hotels edc las vegas edc las vegas hotels.use firechat people edc dont need internet connect friends.anyone extra space trump hotel willing pay part room feel friends good trust worthy guy inbox.lol everyone thinks lyft drivers unless car certified city cpcn back bumper illegal pay strangers rides edc rides free charge guarantee people major risk getting car someone random like multiple agencies town still offering good deals round trip get scammed unlicensed drivers use code discounted lyft rides https lyft com ici scott ph.anyone selling ga day tix.hey everyone want spuce yoyr adventure click link try inofficial fan made scavenger hunt edc set thru goosechase app apple android.iso two day tickets comment message let know price.burn tan.https lyft com ici scott ph.good fam checking room extra queen bed grabs friday monday message details thanks safe travels see edc.need day passes cash ready local well.hey guys really bad breakup ex girlfriend heart ripped apart darkest place ever killing excitement edc waiting come years coming alone celebrate st none friends saved looking get trashed anything wondering willing meet edc feel damn lonely sorry sounds pathetic.looking ga passes.anyone one ticket today.sudden desperate need audio cable headphone aux red white audio buy one vegas staying mainstreet station.one ticket saturday sunday shoot offer interested.make apothecary shoppe edc stop spin win free prizes could include free eighth going weekend long pm edc edclasvegas medicalmarijuana.vegas need ga willing pay friday plus deposit.hi linq shuttle pass looking trade lvv shuttle pass.people stupid talking asking drugs bunch rookies.first time going edc bad security stash boxers good.selling kandi kandi supplies bunch string beads bags box wx lx half full beads made kandi insomniac bag rave hood goes knees everything pictures pm gunna try one last time message interested mickey minnie part sowwy.unregistered vip urbana oh unregistered vip ticket.day pass today home girls ticket fell let know much.dust masks allowed seriously.promoters selling tickets encore beach club today.need ga tickets.flying san antonio today.anyone selling village shuttle pass.anyone still selling day pass maybe even day pass pretty reasonable price lmk.need vegas village pass please help.driving san diego las vegas today friday leaving around pm hit interested room people comfortably depending amount stuff person bring might able pull rd seat stops total.trade ga wristbands vip.anyone need sunday ticket.anyone selling ticket sunday pm pls.anyone wanna tao pool party today.anyone coming san diego orange county extra room car room shit fell ended going home still want go throw cash.wants party golden state tonight vegas hit.need two sunday wristbands please.link tracks drive.interested buying saturday sunday wristband cant make pls message meet vegas.selling two sunday passes msg info.sister wants wear flip flops edc never bad idea right back please edit wants know wearing uggs.driving los angeles las vegas today around noon meet crew oc point picking benz friendly fun chill guy spots open.edc family thing shuttle busses set pick times set places show time set pick spot long walk staying partying get screwed get money back plus leave edc leave full people gotta wait forever set times places come whenever want pick wherever want fuck around get back right hotel part family peace mind.hey fam homie looking extra ppl wanna want save extra hundred dollors hotel share room staying arizona charlie boulder whole weekend hit asap dm interested.girlfriend make edc sunday anyone interested selling sunday tickets shuttle tickets las vegas village want tickets two shuttle passes sunday.tag find spotted ravers way.parties lmao boys supa cool cute tho oscar.home stretch might time kaskade gotta cut another hour though.beyond.free rider credit lyft https www lyft com invite edcvip first time riders lyft like uber.hit free entry omnia night club.open spot tropicana payed make offer.three l bottles cost allset.looking edc day pass.quickest way edc right screw hours traffic.looking day general admission anyone got extra one message plz.rave bae find give unlimited shoulder rides massages.fun safe everyone.friends showed vegas need tickets days call name price.hello everyone wanted share experience edclv noob never gone rave festival edc first come fully prepared forgetting one essential things bring festival dessert would water camel backpack buy insomniac refillable bottle uncomfortable carry along also empty often waited friend next merch booth guy came asking needed water backpack first thought joking handed skipped away maybe high fuck idk still awesome thing course made day happy knew overheat worry leaving set midway wanted share partially feel like want thank way post since thank want remind guys selfless goes long way let actually plur today edc carry along attitude towards others anytime year.hey guys really need help quite literally stranded venetian casino long story short booked hotel someone turned scam looking place crash weekend atleast tonight humble people space room love split please hit thanks pull couches work.late night homies.bitch.whats good fam still got extra queen bed hotel room checking tomorrow afternoon message deets see edc.hand ga.looking sunday pass friend.making bracelets airport wait flight vegas edc come.know late homie looking friday ticket saturday hmu pls.need day vip ga unregistered willing pay meet vegas know last minute someone please give deal looking days.going hard group crew partying tonight anyone wants join whatsapp world post need dance ass.need ga tickets asap make best offer la sfv area mobile plur.anyone extra room ride edc.anyone leaving vegas monday california.hard rock hotel.let go tonight.last edc wristbands day ga available info inbox.drivin right jct stuck traffic admiring beautiful.need ride well need ticket willing trade free ride la back ga vip weekend pass ford fusion se fit extra ppl tents plur alllove letsgoooo.buying shuttle passes linq dropping already vegas.need buy friday ticket anyone got one.vip las vegas nv vip edc ticket days best offer.kaskade ticket san antonio tx kaskade ticket sale text.cool walk non open bottles lol.buy registered ticket way change registered ticket name trying get scammed.hitting omnia.staying stratosphere wants meet.selling ga tickets...neongarden techno nocturnalrebound.anyone wanna share ride airport tropicana like vegas time.selling one sunday wristband sunday mgm shuttle.selling two general admission tickets unregistered unfortunately brother got bad car accident able make anymore trying recoup money passes overnighted guaranteed vegas get sold today asking tickets taking huge loss need money soon possible help would appreciated venmo paypal cashapp.anyone crazy lol coming chicago least making progress.vote new remix fam.stranded vegas need place stay literally one night please help.cosmo.hosting table marquee tonight dj galantis hit want come pm.anyone spots left car vegas leaving friday morning san diego orange county willing pitch gas spreading good vibes let know please.extra linq shuttles around.anyone looking day ga extra ticket buyer back last minute vegas meet asking.selling two three day edc tickets hmu looking.saturday sunday pass need get wrist tomorrow comment message looking rave.anyone needing vip ticket willing trade ga x ga cash.got bracelet register met vegas lmk.search day general admission free.tag met edmdfw texasplur edclv.looking saturday tickets cash ready.know let take gatorades shuttles.anyone damn ga ticket last person bail.looking day edc tix meet vegas.extra day shuttle pass linq hit interested.weekend going fire.vip ticket selling vip ticket obo luxor.pregame penthouse hotel top monte carlo party bus encore beach club pool parties weekend tonight pregame party bus whatever let go hard fuck.anyone currently mgm shuttle pass wants trade linq willing trade mgm linq two us mgm linq want try shuttle.shuttle pass san antonio tx selling shuttle pass las vegas village queens freemount street.think going edc lol.luxor hotle casino.anyone needs ticket please messenger buy offer obo vegas already please message much love see electric sky https facebook com story php story fbid id.st time edc sure expect super excited ready dive meet new people awesome experience.vegas already move tonight see tomorrow.selling edc day pass houston area let know interested make selling cheaper purchase price hopefully someone enjoy.day general dallas tx one edc day general admission ticket sale paid taxes fees meet vegas friday morning.wants party armin van buuren tonight.please come check us.looking day passes vegas friday morning dm.edc wy.day vip tickets urbana oh day vip tickets.rave app meet tonight meet beautiful edc ravers.day ticket las vegas nv selling day ticket obo.made.palazzo cash day ticket register prefer vip pm.see guys soon.vegas need shuttle passes las vegas village pay pick please message.good size hydration pack edc.babe dont speak section bae poop danish.looking two sunday tickets.buying sat sun ticket.need pass anyone selling one.hey guys solo edclv anyone link parties resort staying dm good plur vibes please.dj frequent flyer club airport.bus passes houston tx selling bus passes sunday cash meet vegas monkey business please hmu.get single day passes gates need know homie.whose staying monte carlo meet.vegas arrived deep inside.anyone want go bassrush pool pary two spear tickets.made safely canada hello vegas.much love road.edc shuttle pass leaving lv village lot across luxor selling message.looking two sunday passes help catch last day insanity.need ga day pass.anyone selling basscon pool party tickets.day starwars day random camo day captain america.almost ready depart vancouver bc power banks pain reliever hopefully help end every night.limomafia callbackkingofthewest.dimitri vegas like mike encore beach club today male ga happening right.text free entry guestlists table inquiries phone number also whatsapp free drinks ladies available events marqueepromoter marqueelv marqueedayclub edclv edc vegas vegasguestlist vegashost vegaspromoter.looking day ticket.beyound hit guys see today.mine crew first time going electric daisy carnival edc also first time making kandi weekend one books.anybody going encore beach club morning.vip ticket atlanta ga vip ticket sale obo message asap meet vegas.selling one ga bassrush pool party ticket pdf ticket.whos rocking rehab bassrush poolparyy.running late convoy.looking one day ticket friday.kaskade tonight hakkasan nightclub oliver helden jewel nightclub armin ban buren omnia nightclub free entry guestlist guys girls hmu pm pm reservations male promotions manager hakkasangroup instagram jerryngo viplasvegas hakkasan l omnia l jewel l wet republic ultra pool l.selling bassrush massive ticket tonight plans changed e ticket.ga wristband albuquerque nm one ga wristband best offer arriving vegas today let know interested.pool party luxor hotel casino selling beyond male admission pool party ticket hit.selling sunday vip wristbands dm.ravers lasvegas edcweek fattuesday.sharejumber mixedclisten.eles seatac airport waiting pm flight.dose anyone shuttle passes sale las vegas village need two also need one ticket bassrush massive tonight way vegas pick please message.ga las vegas nv ga days sale ga sunday sale ea pair please pm require deposit hold ticket pay remainding meet venmo payapl cash.first time going idk expect like techno heard drumcode gonna fuck yeah excited actually sell alcohol bottles like cups shit.attention edc peeps come pregame every morning cheap botomless mimosas bloody marys hard rock cafe right strip next big coca cola bottle every day till noon mention name male work come trade kandi.hardwell tonight hakkasan nightclub jauz jewel nightclub lil jon oak nightclub free entry guestlist guys girls hmu pm pm reservations male promotions manager hakkasangroup instagram jerryngo viplasvegas hakkasan l omnia l jewel l wet republic ultra pool l.needs place stay lmk.weekend going lit.staying trump.got one day vip sale inbox info.selling day pass day parking pass hmu.firm meet las vegas.sale firm.pool party today.wanted shuttle pass mandalay bay days prefer pm premier.anyone selling shuttle pass festival grounds.anyone selling friday pass go meet vegas.meanwhile waiting denver wants play barbie hair salon thats.cause thursday las vegas still plans come join us free screening electric heart movie pm united artists showcase.hope guys gals safe travels whether planes trains automobiles much love lets rage electric sky see..anyone flying london uk.anybody got fire ass suggestions places eat las vegas staying bluegreen club pretty close strip.paul oakenfolds dad flight dj austin powers.pass henderson nv hey girl ended able go weekend trying help sell ticket day pass general admission located already vegas asking obo feel free offer pm facebook thanks reading.selling day passs meet ups vegas.anyone looking day pass.question mobile service speedway.day ga ticket hmu interested.first plane second go edc come.airport waiting gate trip officially underway see guys hrs.right popcorn guy orville redenbacher.rolling solo trying link drink blaze hmu snapchat kushtv.ready.let link.anyone know time borgore playing.anyone spot hotel tonight.anyone going dimitri vegas like mike today going dolo cause friends bass heads lol.one ok take.looking las vegas village shuttle pass.day edc pass lv village shuttle pass across luxor meet vegas tomorrow sell together hmu.mosquitoes vegas tired getting eaten alive need know need bring bug spray.hi im looking ride colorado edc anyone know anyone room would really appreciate money gas ill edc youd like meet.comment like share ill group chat since nobody making one.allright edc ravers lets post outfits edc day starwars edcday.looking mgm grand shuttle pass anybody one.shuttles passes las vegas nv selling shuttles passes mgm.day ga ticket selling day ga ticket la area deliver.attention everyone planning make youtube video first trip edc going sunday wanted record much possible meeting new people people costumes totems candy dancing light shows anything else catches eye leave snapchat name bottom plan record interview people serious know little weird talk snapchat maybe ideas help edclv snapchat bomb com.anybody gonna hakkasan tonight.sure use lyft code hookup rides.anyone know play kittys.help need tickets hardwell tonight.best fran got extra queen bed hotel room ellis island conveniently located near strip cheap bomb ass sirloin steak breakfast beers restaurant lobby free parking also ride us festival back msg interested.hi would like buy two tickets friday night text message two would like sell purchase new ones insomniac take sat sunday ones.price drinks like.need linq shuttle pass.anybody going jauz tonight.see everyone sat sun.anyone bay area going edc leaving friday night sat morning room need ride last min trip located san jose.alright guys weeks emailing insomniac trying spots routes talking drivers finally came efficient way transport people edc party busses taxi uber upwards get speedway paying sit car traffic party busses pick times right pm pm pm way back till buy way one way please left without ride people try take advantage giving wristbands offices selling busses strip well first come first serve busses byob provide water snacks rule bus good vibes one way ticket ways pretty cheap ride party bus please send text many people group time would like also guys interested pool crawl club crawl also selling fast lets crazy weekend.know edc couple days may crazy ask anyone happen extra room one need find place stay.bought shuttle leaving village would like switch leaving linq would anyone know possible thanks.know las vegas recently passed new recreational marijuana laws mean okay use medicine inside edc policy.come trade kandi.cosmo let party together also anyone got link dreamstate pool party tickets get huge group get free open bar hit.looking buy day pass located vegas good.anyone north county san diego heading toward one n edc shuttle pick.anybody extra spot car leaving friday saturday la ca.need friday saturday tickets please meet vegas need good deal pleaseeeeeee days highly priced.sup fam need spot crash edc hotel room extra queen bed grabs message deets stay lit see edc.ugh hrs till inside las vegas.bassheads wanna kick.come trade kandi im work hard rock cafe strip outside cant miss.vacation begun finally.soooooo one even started packing yet haha.iconic.gotta pretend working next hours lmao.enjoying edc week vacation.parties popping tonight house sweets etc.already vegas turning.getting crazy vegas yet lol.pick pair electro wings time weekend stay lit electric sky https www amazon com light green fairy electro wings dp b xnw vms ref sr ie utf qid sr keywords electro wings.ride credit download lyft app using referral link https www lyft com john affordable ride town terms apply.anyone staying flamingo.looking ticket friday night anyone one sale.local lyft driver las vegas little help tip edc providing free shuttles different spots city strip google case also anyone lyft account hit link get around vegas low importantly safe ride credit download lyft app using referral link https www lyft com esteban affordable ride town terms apply.alright guys cliche post time first time going edc want meet fams want get huge group pic.meetups mgm grand.wallet stolen monday shitty know village shuttle passes emailed insomniac getting replacement said come mandalay bay get new wtistbands know shuttle pass wristband curious anyone else experiences past like.selling day wristband located south bay hit fam.someone ticket registered make sure scam.mind ready edc body bodyyyyyy telling nooooo poolpartynotready cakeme friedchicken.got two day pass available hmu met vegas.anybody got vip ticket sale lmk asap cash.anyone selling edc tickets.looking saturday ticket.need linq shuttle pass.man flying today landing midnight staring clock work.lol.hi hope meat u.troll trailer time roast ravetrain edition sexy time.driving edclv sacramento leavimaleorrow night around midnight anyone looking ride pm.anyone ticket sell adventure club drai beachclub sunday.still looking two wristbands sunday lmk.someone tickets sale friday need.girl looking cheap place crash fri sat night please pm w info.stores strip sell camelbaks.fkn employer trying take vacation away day starts cause accidentally approved many people shit fault dumbass looked plus first get approved bitch got fck lmao.landed edc week starts us.coming wet republic today beyond.listen edc mix huff n puff np soundcloud.tonight edc week come party underground get loaded psytrance event link www fb com events.people flying seattle looking expand circles edc make friends stay different hotels stay tropicana let know wanna fun together.needs ride oc vegas.vip edc pass phoenix az extra vip edc pass sale obo.listen edc mix huff n puff np soundcloud.heading airport vegas come.everyone best friend extra bed available hotel room anyone needs place crash looking people join us strip ellis island really bomb cheap steak breakfast beers day also room car la area hit ride message details pricing thank see edc.need one two tickets saturday sunday let know anyone selling edcvegas.selling one day pass without shuttle message price.anyone las vegas festival grounds shuttle friday.anybody want come free screening electric heart movie tomorrow las vegas featuring music armin van buuren dillon francis dash berlin wandw many trailer https youtu qdlwupixrxu.hey everyone get around safe cheap las vegas lyft pick airport get edc hit link take enjoy las vegas safely https www lyft com esteban affordable ride town terms apply.bedroom house near freemont street edc party girls welcome.meowww east coasters chillin.leaving vegas tonight add snapchat let rage night away.east coast squad nj ny flying tomorrow energy starting kick.super towing llc cpcn.looking saturday ticket paypal ready.girls since powder makeup thing bring unsealed found low priced items dollar general appear decent cheap enough buy multiples brought sealed day good enough touch pencil eyeliners lipgloss etc idea.make key guys.leave tomorrow morning canada excited wait.wave happiness going unknown came way work morning played deadmau closing ultra last year bring baby bring.heads free merch jump.know one drop fire confetti one.anyone selling edc ticket.hey everyone kinda new kinda music still need help set times people wanna see really fan whole head banging music maybe point night going kinda stages part want plan good day way know wat djs play best sets interest know guys know lot w someone please help looking like trance anything like.hey peeps anyone got good recommendations aussie blokes travelling la vegas tomoz party buses heading way.anyone going friday buy wristband get back saturday morning.anyone leaving columbus indianapolis st louis kansas city areas tonight today might room two people eesha happy vibes greyhound bus hours behind schedule puts us vegas middle night would love love help.anyone south end vegas catch ride sunday.made girl bra u ladies think.edc sunday wristbands w shuttle west covina ca selling sunday wristbands w shuttle festival lot located cal area going first days keep wristbands clean loose easy access sunday meet locally vegas.selling one shuttle pass las vegas village car pooling instead rave fam.need linq shuttle pass.last call someone needs day ga
    


```python
female_posts_as_text = clean_female_posts.str.cat(sep=".")
print(female_posts_as_text)
```

    insomniac released statement man death.know kaskade performed edc yesterday effff.https www facebook com jushonti giberson posts st edc crazy shit happens.blast link day called zelda tho.one best time life see next year electronic sky.lost item edc unable make contact lost found department please e mail lostandfound insomniac com auto respond claim form fill lost found booth still open open tuesday pm actively checking claim forms inventory arranging get items returned know place post feel free copy paste.many people sick head cold edc next year taking massive immune boosters several months festival hope others better right unfortunately everything happens vegas stays ugh.someone vegas help get weed wax.ok watch lmao.really sorry people good time stellar time dinner next armin friday mr chow friends met aoki saturday night anyone else good time weekend love see wings come say hello kandi give away.everybody saying edc sucked year learn patience understand desert hot eat full meal every day even hungry force eat keep water full times remember actually drink see want see get stuck trying stay group friends came alone edc year truly greatest weekend life anybody wants chat personally happy share tips.anyone still vegas wanna ceremonial burning shuttle passes kidding seriously.whoever kid neon garden wish coukd dance like lol gettttt ittt.hey guys give warning used atm next skydeck near kinetic field check account someone stole want get crap stolen rookie mistake know lesson learned spread word everyone besides b first year edc amazing wait see electric sky next year.curiosity anyone take ubers weekend much strip surcharges.fucking ac shuttle bus.anyone buy hyte nyc tickets yet time single tickets code hrbrloyal valid till pm tomorrow pk deal limited time offer feel free use share personal link http bit ly hyte sara https youtu yp wgkquxfk.note say see one mates fitting overdose stand watch get help towards end andy c set walking saw guy seizure ground overdose friends stood watched knowing immediately went got boyfriend first aid trained also police police kept trying lift floor restrain rucksack pointed guy fit still going blue lack oxygen tried give cpr without checking breathing first could died police really clue took got mouth opened biting tongue teeth fit blocking airways squeezed cheeks pulled tongue pressed get air started breathing seconds colour started coming back face flip side sicked hands waited come round police eventually carried see medic trying say see something happening similar wait send one friends help check friends airways clear get mouth open get tongue free push get air friend guy lucky passing really scary girlfriend mortified stand around watch way wanted end awesome time andy c set hoping involved ok message might get.stranded las vegas looking ride back monterey santa cruz people either tonight tomorrow chip gas etc.anyone zedd omnia tonight.somebody going las vegas la tomorrow wants share ride.wsa first heard battery reconding program thought fake start using program step step reconditioned dead old batteries surprised save money details go.tips next years edc u leave speedway time everyone else complain long shuttle car lines leave earlier u mind waiting long lines fuck stay till end cause u vip mean ur one people vip well expect get special treatment buy tickets drugs strangers bring ur shit home complain heat incase u forgot vegas get hot even night shuttle lines bad year atleast moving shuttle rides boring u let use time meet new people play games sing songs u see someone need say something keep hydrated gonna atleast one two hip hop artist u like dj playing seen dj khaled way said edc year dope aside little things like ankle twisting day one way speedway good edc.night.bought two tickets yesterday able get edc last night boyfriend enjoy decided sell guess tickets scan today tried contact bf answer us bucks plus money return people sold shitty kind people like honestly sure parents raise animal thanks jessica rodriguez fucking us.time life married best friend went festival dreams thank baby edc back.monxx b b herobust.reason everyone selling wristband sunday outta loop.kids learned lesson buying tickets strangers lol shady fucken people.needs ride back cali giving rides back per person.even cute marshmellow outfit fits wasteland gammer played thank much guys people appreciates outfit im overwhelmed see next year electric sky hoping deadmau next year.vip wristband circus circus las vegas hotel casino vip wristband festival grounds shuttle pass obo circus circus.day lets go.need wristbands.got vip wristbands linq shuttle sale.entire group decided go last day selling first come first serve.edc day cya marshmello.thank amazing people gave water air sorry making shuttle stop ended hospital much love amazing peeps helped super sorry b c girl handle heat.looking three edc tickets tonight.one sale.selling one sunday ticket meet strip.saw group called edc fp seems really cool anyone knows contact.hi everyone part fb group called las vegas painted rocks painted edc inspired rocks would like hide guys find vegas also meet trade kandi name fave song whatever still painting anyone requests ideas etc.looking sunday wristbands trying let friends experience edc first time help us please.looking one sunday wristband plz thx.anyone selling sunday wristband need one.edc shuttle passes edc shuttle passes sold village line best offer.need tickets edc sunday.much uber edc fuck shuttles.edc juggling taxi hat edc monday night casino jokesters comedy club june keith lyle dee brooks.text edc ticket sale en route speedway text price asap.bought two tickets friday able get edc friday issue boyfriend enjoy decided sell tickets guess tickets scan saturday tried contact bf answer us apparently reported tickets stolen able get new ones bucks plus money return people sold shitty kind people like honestly sure parents raise animal thanks jessica rodrigues fucking us please help spam instagram.vip tickets vancouver wa vip tickets.tickets hermosa beach ca edc tickets sale tonight wants em monte carlo hotel right mgm paris hotel.looking pick tickets round pm.looking sunday passes friends behind encore.dm.looking sunday tickets.anyone near luxor selling wristbands together looking shuttle bus passes.selling shuttle passes today sunday hmu.molly.one linq shuttle pass selling one linq shuttle pass.edc sunday pass village shuttle schiller park il sunday pass sale vegas village shuttle pass meet delano.looking edc tickets today.sold passes chickasha ok two passes tonight sunday also shuttle passes pm plz.adventure club venetian las vegas selling adventure club hmu.selling one sunday wristband village lot shuttle pass.cash money los angeles ca bucks cash money one fancy edc bracelets hmu.looking sunday edc wristbands.selling wristbands today sunday day edc hmu interested.need two edc tickets tonight able pick.looking edc tickets.two sunday wristbands arlington tx selling two sunday wristbands.still sunday pass linq shuttle pass go hmu.two linq shuttle passes free cosmo right anyone interesting.still selling sunday ticket please message intrested.selling two sunday ga pass shuttle pass message interested.selling ticket tonight sore dancing think another night lol.anybody selling sunday passes.sale sunday ga linq shuttle.anyone selling tickets need.mgm shuttle pass san diego ca selling mgm shuttle pass village shuttle pass.selling mgm shuttle passes las vegas village shuttle pass message interested.mgm shuttle pass kansas city mo selling mgm shuttle pass best offer.looking sunday passes meet within one hour.premier parking pass got back room one hour last night.selling village shuttle pass anyone interested.plan changed leave vegas earlier selling cheap tickets rd day admission shuttle bus las vegas festival ground ticket per person shuttle per person please pm interested.look shuttle passes premire parking pass.sold edc tickets free green bay wi selling edc tickets las vegas village shuttle pass pm interested staying excalibur.good morning looking buy sunday wristbands pick cash hand anywhere vegas around pm let know.looking edc tickets.sold unregistered madison wi selling unregistered message.looking sunday tickets.edc amazing experience looking forward going day husband going solo selling vip wristband mom watch kids anymore best offer takes staying mirage meet room know bogus sale.need ticket.hey guys looking passes today.mgm grand shuttle pass federal way wa selling mgm grand shuttle pass one.peace love unity respect share love.sunday pass las vegas strip selling sunday pass gotta come pick.one sunday ticket sacramento ca one sunday ticket sale.need sunday ticket.looking pass tonight selling much thanks.selling sunday pass anyone interested let know.looking sunday pass someone could sell ended giving someone else soooo help.day sunday passes edc kinetic field las vegas motorspeedway.looking sunday bands shuttle passes trust guys say terrible message plz cash hand.need one sunday ticket lv festival grounds shuttle.one sunday pass available shuttle pass available pick aria.brazil confirmed.village shuttle pass.best way loosen wristband literally caught dog trying eat mine pulled way closed.ga wristband romulus mi selling ga wristband sunday las vegas village shuttle pass box stuff used pick flamingo hotel cash.one gold one ga tomorrow fully working problems pm offers total.lmaooo.selling one sunday edc pass requires deposit meet anywhere strip.edc tickets selling edc tickets obo lmk.driving vegas la soon wanna drive alone willing pick close dm.looking sunday pass anyone one sell let know thanks everyone.need sunday ticket.anyone selling saturday tickets.anyone flown edc via helicopter.need ga tickits saturday sunday asap dm let know please.sunday edc ticket san gabriel ca sunday edc ticket dm please.need two sunday tickets.amybody going hakassan tonight.attentonshuttlebuswaiters lol bc yall aint riding yall waiting extremely long lines get lyft rol rideoutloud team people pack car like sardines split bill get day.wtf happened paul oakenfold.found iphone yesterday message give lost found tonight.traffic right.lost pass take uber tonight go friends going split drive selling mgm shuttle passes together obo leaving tonight meet asap anyone interested.looking sunday las vegas village shuttle pass.one shuttle ticket downtown sale.selling unregistered new weekend ga saturday sunday meet strip afternoon dm.ga pass downtown shuttle sale.first edc want explore vegas little tonight anyone wants go explore free seats good vibes good music devil lettuce hmu want join let form tiny rave fam seats fill delete post wonderful day.selling one wristband interested.looking two day wristbands.anyone eyelash glue bringing lol real question.insomnia needs get uber situation check lane ubers come pick people serioisly legit waited hour uber driver min away.looking two day wristbands.selling two sunday edc tickets.anyone selling shuttle pass.looking passes.parking sucks saw many traffic control people directing heavy traffic.two day shuttle passes village two day passes mgm sale meet mgm pm.need one ticket linq shuttle pass anyone.anyone shuttle pass.need sat sun band shuttle pass lmk asap.looking buy sunday edc ticket meet strip downtown sunday morning afternoon message lowest asking price thanks.need wristbands remaining days help bish.shuttle charleston sc got sold shuttle passes linq asking.looking sell ga bracelet tonight meet back tomorrow go sunday located marriott grand chateau behind planet hollywood message best offer cash prove wristband registered person.need day passes also consider sat sun passes hmu.first line shuttle hell yeah.ga vancouver british columbia selling ga sunday edc wristband pick mirage fb message thanks.selling mgm shuttle pass.edc sunday passes vancouver british columbia two edc sunday passes sale meet around mirage sunday afternoon asking thanks.linq shuttle passes vancouver british columbia hi linq shuttle passes sat sun asking want buy one meet around mirage anytime pm thanks.bring makeup also extra shirts getting married white shirts probably wont white time lol.parking pass santa ana ca selling parking pass rest weekend.looking sunday pass unregistered less dm.hungry place guys recommend eat strip.selling shuttle pass festival grounds msg edc today meet signal sketchy may take time get back.two village lot shuttle passes spring valley nevada.parking pass santa ana ca selling parking pass rest weekend.selling sunday ticket lync shuttle pass message know interested obo.anyone take free park ride shuttle last night wanna see better driving.selling unregistered edc ticket saturday sunday also selling mgm shuttle pass.rehab beach club audiotistic pool party featuring k madeon seven lions trippy turtle sunday june day bed starting around head drinks bucket trying save everyone cash fun pm.ever u leave edc dj wanted see play planned hopefully tomorrow better day roommates want leave early.someone please tell taxi drop shuttle bus.chain smokers ebc june th encore beach club.want know way sell two wristbands tonight back go tomorrow night without getting scammed ensure get wristband back tomorrow tia.saturday wristbands arlington tx selling saturday wristbands.edc sat sunday santa ana ca selling edc sat sunday premier parking.edc pass tonight tomorrow shuttle pass village luxor message.selling sunday edc ticket obo las vegas village shuttle meet aria sunday pm.got vegas.sunday wrist band linq shuttle meet flamingo hotel.still iso ticket tonight sun.anyone tickets today tomorrow sale.today.anyone willing trade linq shuttle pass village lot shuttle pass sunday.anyone selling edc ticket.need tickets tonight saturday plz pm.anyone selling edc ticket.sale electric daisy carnival edc vip saturday sunday wristband hmu.selling mgm grand shuttle pass.u get day one experience dont want drive today ones uber much back.desperately looking sunday wristband help.selling two saturday sunday tickets sell msg info.shuttle pass concord ca shuttle pass las vegas festival lot near circus circus roundtrip sat sunday obo first pick gets leaving back home today asap.partying front sparks last night send message tried adding snapchat left stage work booooo dodgy snapchat hope everyone killer first night.need ticket saturday sunday shuttle pass would plus.friend got vegas pm bus still trying get new zealand expect anything insane feeling sorry wondering anyone wants meet tomorrow heading would love improve sad start edc.selling sat sun tickets.sunday wrist bands festival shuttle passes.looking wristbands today.lost wallet found please message need id fly back care cash thanks.looking saturday ticket asap.need tickets tonight tomorrow message asap.looking saturday ticket.looking saturday sunday passes pls message.need day ticket ga anyone.anyone know call village shuttle busses.amount trash people threw ground yesterday made sad please treat edc respect deserves keep beautiful.anyone chance uber taxi home show wondering worth escaping miserable shuttle lines tomorrow.ga edc shuttle concord ca selling one ga need shuttle passes shuttle passes departing near circus circus roundtrip sat sun.looking sat sun wristband shuttle av circus circus rn.left candy girls bathrooms place last night flipping hot hope lovely lady found b c canada xoxoxo plur love.need saturday sunday bands please cash hand.looking saturday sunday wristbands message.many u fit hours.looking saturday pass buy sunday.saturday taking picture flag boombox cartel come bring flag take picture mexico edc cheerleader.edc lv anyone went galantis set recorded proposal please post thanks happybridetobe.reshare.one ga pass concord ca selling one ga pass need shuttle passes shuttle passes departing near circus circus roundtrip sat sun.hey guys lost edc wristband anyone wristband buy linq.line like village.anyone heading vegas saturday night go sunday la wanna drive alone.hey guys see us dino suits use edcdinos instagram.looking friday tikets cash hand speedway premier parking lot please message.really cool.iso still looking two sunday ga pm.moment get scanner ticket get green light woohoo wait meet lovely people.getting ready excited drive edc remember download rideshare apps use following codes free rides new users use codes lyft battlebornlyft uber zdrg p sue existing lyft users use code lyft rideedc get existing users per ride next two trips codes used multiple times everyone group download apps use codes everyone used new user codes use existing lyft code rides close free one wants deal cops play safe take uber lyft edc electricdaisycarnival edc edcvegas.arriving vegas edc ya coming.hello festival lovers first edc tomorrowland ade asot umf post one coming edc planing drink take drugs know rear thing lol planing arrive edc car save traffic shuttles also premiere parking give car passengers fast entering venue offer willing take max ravers enjoy free transportation fast entering venue need drive way speedway really hoping find kinda ravers r interested send message.looking saturday sunday pass pm.hey guys kinda need help first time edc friends got linq shuttle pass l wondering supposed go get shuttle bus time.edc get made way gorgeous hair makeup breonca ganae book cover basics bre full face makeup click link bio business page brevamped head website vegaswedding brevamped lasvegasweddings makeup makeupartist vegas vegasnightlife vegasmua lasvegasweddings prommakeupartist awildhairvegasmakeupartist.edc ga ticket edc ga ticket sale.bff looking saturday ticket also shuttle pass sat sun chance anyone selling one.check free ride codes new existing users lyft uber get edc amazing complimentary amenities ravers plur.friend selling made whisky dm interested smooth let know sample look online called gold bar.looking devils lettuce anybody know find hmu.hand wrist band go.every club strip dress code.need two shuttle passes pleeeease.gorgeous ladies need help staying circuscircus would help putting eyelashes glue would super awesome.looking shuttle pass mgm.mandalay bay call.looking day ticket pm pls.anyone selling day pass.wants come n play beer pong venetian.get painted edc.uber venue cancun resort long plan take first time edc still couple hrs really hoping miss ghastly.need one day ga give best guys last person bailed us.looking friday saturday tickets day day reasonable prices vegas pick.big friends group check festiie smart wristband.hey guys anyone know anyone selling edc tickets.got shuttle passes downtown sale.x edc shuttle passes downtown pasadena ca x edc shuttle passes downtown sale.selling wristband ga san diego area message interested also downtown shuttle pass.selling three day general admission edc las vegas need get rid.anyone know might could find buy big red purple black bow strip.selling sunday wristband located westin.looking one ticket tonight sell three ticket sunday edc.looking vip primer parking pass msg please.insomaniac ticket sales office strip semi scammed need buy ticket directly source.anybody needs mgm shuttle pass.guys tell luxor iheart raves pop many entrances.choose stay circus circus attitude employees elevators late change reservations trust.else riding solo edc.anyone staying linq little rubber bands pay.selling town shuttle pass willing trade village pass.anyone selling two shuttle passes need two.selling town shuttle pass willing trade village pass.armin amazing last night.souls need place stay th room available paid already bit staying elsewhere need least get money back lose tremendous gain especially prices rooms th miss hmu asap ps booking confirmation.shuttle pass sale village lot.encore lit tag went.need trade downtown shuttle linq shuttle lv festival grounds shuttle.staying circus circus wants pregame.best time lineup shuttle.edc day reminder everyone chooses party influence finds need care medical crew edc high end basically built mini emergency room site need care arrested consume age took kicked festival pay onsite treatment even insurance country feel dizzy head achey nauseated dry mouth sticky may dehydrated heat exhaustion take lightly especially plan go influence unsure go medical need kind help snag ground control staff member purple shirt angel wings back hard time oasis counselors specifically talk someone private fun safe.anyone know call shuttle pass across luxor.really need hydration pack.bff looking day ga last minute trip anyone selling.free dolewhip wristband regret going.looking edc ticket vegas pick lmk price.solo ladies would like join group girls lmk also give ride seats car stay safe girls.anyone selling friday pass.still spot big bus rv park hitch post mile away edc dm looking place stay.friends need one saturday sat sunday pass please haalllpp.uber taxi airport.staying platinum lets meet.anyone meet inside edc group safety meeting.anyone know shuttles depart soon line.looking edc ticket willing come throw best offer.forget download free uber lyft apps get festival driving vegas drag especially partying favors use codes free rides first time riders lyft battlebornlyft uber zdrg p sue everyone download apps use codes get free rides.looking sunday tickets.making kandi plane sweet lady philippines asked make one afterwards gave us chocolate.day ga edc granite bay ca selling day ga edc ticket obo picked call pictures everything need photo id credit card used purchase pick forward emails well including confirmation order additional instructions pick email front gate tickets confirming someone else successfully pick ticket number case anything fishy happens.hello vip sunday pass edc sale hmu interested vegas.hrs way las vegas mx luis eduardo.totally messed reservation circus circus thought check friday instead thursday tried call caught late think okay.last minnnnnnnn need friday tix.someone please tell brother law get us passport card dl expired.http www thenowreport vegas article local entertainment weekend heat causes concern edc.yassss.xs see chainsmoker going.consular id acceptable edc enter consume alcohol.general admission registered meet vegas.hey roommate went iheartraves emazing lights pop store earlier tonight dropped anyone found cash lying ground outside pop store please let know money left.got day pass edc unregistered comes box better offer hit interested.last minutee edc anyone wristband could buy.anyone selling linq shuttle passes last minute shuttle passes.day ga tickets pick las vegas transfer name receive payment.ga tickets temecula ca day ga tickets sale meet vegas tickets cash paypal.anyone going hakkasan night club tonight see kaskade.made vegas anyone going marshmello tonight intrigue trying go sure two friends wanting.looking las vegas village shuttle pass cheap please paula twerk whichever prefer pm.selling ticket saturday sunday.guys bringing meds lol.anyone need chainsmokers tix tonight sell x paid.looking lv village shuttle passes please need.want see shenanigans going one n edc trip follow us snap.females going edc alone.looking saturday ticket.good evening fellow edc attenders going attending edc first time year happen hotel room set posting request someone graciously allow use restroom even one days mere minutes shower prepare day also driving vegas many friends attending edc likely provide restroom swiftly use three days attending friday saturday sunday part experience trying engage socially interacting completely random people feeling sense community support one another likely find find last resort hope someone kind enough allow short time would mean world right fact somewhere thank taking time read post arriving las vegas within hours would appreciate available restroom anytime convenience.last shift edc officially time finish packing.car ready drive vegas.looking mgm grand shuttle passes.heading edc come visit us sin city blo bar located right strip sure hair makeup glitter glow electric sky pre booking appointments www sincityblobar com.need braider anybody braid girls hair friday extra staying paris plur vibez.brand new b pick holds.let shit else fails drive vegas flights chicago looking rough.chainsmokers tix selling chainsmokers tix tonight.need linq shuttle pass lmk one vegas already.desperately looking day ga ticket shuttle im vegas meet please help get edc.days ga pharr tx selling days ga pass hmu.excited see edc add ig snap belenfls.need one day ga give best guys last person bailed us.imagine weekend going weather..wanted buy something inside apple pay would rather carry around lot cash.ayyyyyeee sls hotel.due unforeseen issues able attend kaskade hakassan tonight drive chicago vegas vip male tickets drink cards sale feel free make offer.let vacation begin.got general admission bucks yes im losing money cus dumb fuck message.delete allowed sharing love fun guys gals.texas help rep edm dfw wait see picture yalls beautiful souls.traffic guys leavimaleorrow morning know expect first edc.edc sat sun tickets sale.anyone travel.anyone selling sunday ticket need one please thank.free drinks day vegas app called cityzen free app strings attached gives free drinks participating locations strip download sign using facebook allow enter promo code free drinks code wc delete leave grab drinks need help pool parties nightclubs dm details.https www facebook com kitchenfunwithmy sons videos.offering custom airbrush spray tans two come bring tent equipment room suite house get edc glow pm comment.fun edc stay traffic trouble.day general one edc day general admission ticket sale obo paid taxes fees.https facebook com story php story fbid id.selling day general pass l area vegas friday noon.anyone know day parties sat sun monday like clubs bunch cool people coming together.ya mero.way vegas see guys kinda traffic past modesto bcoz road construction.selling beyond male admission ticket wet republic today.people getting vegas today making jelious keep telling tomorrow make leave airport fly eastern time.cheap bottle service people hyde drais intrigue bottles taxes gratuity included dm interested wednesday night hyde maria marano thursday night drais dj esco intrigue marshmello.anyone else procrastinate festival outfits like lol hope vegas shopping.anyone else coming philly cityofbrotherlylove.started packing see thursday corazones.staying apartment north strip south speedway really like hitch ride someone get whole epic walk experience ya.ok party.free drinks day vegas app called cityzen free app strings attached gives free drinks participating locations strip download sign using facebook allow enter promo code free drinks code wc delete leave grab drinks need help pool parties nightclubs dm details.really bring gloves edc.anybody needs rides anywhere message local cheap price plus drive minivan take cash credit haha hubby driving well.high guys see u soon.tonight next nuphonic rhythm multi genre warehouse party one room one vibe one night edc begins bday celebration lighting guy jordan koi breakbeats psytrance dnb techhouse trance basshouse futurebass futurehouse https www facebook com events.ride airport heerrreee seeyou lovely souls soon.hi everyone two ga day passes really need get rid linq shuttle pass anyone needs.gooooo.anyone selling premium parking pass please let know serious buyer.bottle service people hyde drais intrigue bottles taxes gratuity included dm interested.kick edc weekend half drinks stop cabaret show lounge inside planethollywood friday come find behind bar major discounted drinks half anyone wearing edc wristbands anyone shows post doors open pm june th need drinks heading nights major event catching shuttle planet hollywood hotel nearby come hang burlesque themed lounge message details cabaret show lounge.know edc week start see around las vegas.sunday intro last year.yall heard thats man.selling two tickets anyone interested.vip ticket brownwood tx selling vip ticket includes overnight shipping message interested.someone wants share ride room car left go de strip somewhere else festival.looking ticket lv village shuttle pass.hello fellow edc fam hope everyone excited ready weekend wanted announce photography collective fifth district going weekend anyone wants pictures creatively designed outfits gladly rates manageable quality service outstanding order forever hold memory awesome weekend would encourage take advantage opportunity feel free look website www fifthdistrict photos examples work interested setting photoshoot us reach us contact fifthdistrict photos lets make edc memorable party.saturday sunday anyone interested.ideas things nice inside festival right camelback filled deodorant brush sunscreen sealed gum sealed portable charger sunglasses.take rechargeable phone chargers first edc.someone wants share ride room car left go festival de strip somewhere else.still ga pass mgm shuttle pass pm interested.hey plur fam looking friday saturday edc tickets thank.selling day ga pass best offer unregistered sf bay area.soclosetohome.shuttle buses tickets free greek quarter melbourne selling shuttle buses tickets las vegas village cheap inbox interested.anyone needing ride sacramento las vegas leaving tonight work making pit stop reno pick friend heading shortly.please stay hydrated especially ur gna b drinking hard take advantage free water stations know lines suck moves fast people camelbaks offer water ppl fans ur much appreciated u see someone dying heat u water share lets look one another dehydration heat exhaustion real u guys last thing u want happening miss day cuz u ddnt wana drink fkn watwr.hey everyone im looking two tickets sunday please private message thanks.surprisingly seen many people posting yet excited see edc top year got dillon porter axwell ingrosso seven lions alesso damn time dillon plus house dj green velvet duke dumont joyryde diablo oliver heldens one last note please everybody spend bit time quantum valley trance fam absolute best.hi beautiful souls extra shuttle pass village lot got free want pay forward give whoever needs live bay area meet vegas first come first serve.anybody selling premier parking pass.allowed bring sealed vape liquid inside.anyone going bassrush headbangers.anyone selling edc tickets.selling day ticket anyone still looking.raversbyravers looking rave edc got got hr long rave edc weekend interested come see weekend work las vegas strip want info comment https www facebook com events ti cl.lgbtq people let party.trying get tickets saturday sunday least sunday please let know prices las vegas.need take n uber lyft use free ride lyft use battlebornlyft uber use zdrg p sue forget try uzurv battleborn free reservation beat wait times pm reserve driver taker pre arranged rides done app app cash rides please text.looking two single day tickets one day ok long two tickets partner going drive uber lyft want spend one night festival two us couple two young children need good weekend year.electronic uber gift cards want bought thinking would use need value selling obo preference google wallet need meet exchange unless feel comfortable.landed anyone space tonight till tomorrow please thank.airport swear luggage weighs like lbs weighed weighs lbs.expensive get nails done vegas go need edc ready.anyone needs day ga pass unregistered bay area hmu obo.tickets selling four female tickets illenium nghtmre liquid pool lounge pm.everyone plans thursday night everyone drop snapchat names instagram mine snap lilianadejesus instagram lilianadejesuss.anyone driving denver vegas.looking ticket sunday anyone selling please pm offering thanks.looking buy two shuttle passes mgm plz message help.hey fam forget pack.anyone tell vibedration vip packs allowed.tickets richmond ca sale two tickets days.anyone recommendation time get bassrush pool party avoid lines difference lines vip thanks.anyone looking vegas pool parties june friday sunday thursday daylight beach club claude von stroke friday laidback luke saturday rehab beach club lau sunday wet republic afrojack pool party tour takes first bar get fucked lol first round shots us open bar hours party games beer pong flip cup play alcohol free head major pool parties drais beachclub daylight beachclub wet republic sapphire rehab free entry line ups head vip general admissions usually sale use link purchase tickets message dm schedule events details.one edc day ga admission strip hey guys one ga day pass sale pm details genuine ticket trustworthy reliable source rip x.vegas nightclubs hookups wednesday monday june hardwell armin van buuren zedd axwell ingrosso afrojack steve aoki steve powers gta esco irie nghtmre nightclub party tour takes first bars get lit first round drinks shots us open bar hours party games beer pong flip cup play alcohol free head major nightclubs hakkasan omnia drais jewel light bank chateau foundation room night swim free entry line ups head vip general admissions usually sale two events use link purchase tickets message dm list dj venues.got outfits figured need shoes prefer boots suggestions.anyone looking vegas pool parties june thursday sunday thursday daylight beach club claude von stroke friday daylight beach club laidback luke saturday rehab beach club lau sunday wet republic afrojack pool party tour takes first bar get lit first round shots us open bar hours beer margaritas shots party games beer pong flip cup play alcohol free head major pool parties drais beachclub daylight beachclub wet republic rehab sapphire free entry line ups head vip general admissions usually sale two events use link purchase tickets message dm schedule events details event link dm also nightclubs message list djs events.hey guys im looking fellow ravers preferably couple join adventure edc im offering ride bay area edc las vegas need staying embassy suites suite sofa bed get room pretty much friends recently bailed left looking people split things smb interested need place stay message thank.shoes dublin ca led shoes women size never use new box fit trying sell meet las vegas thursday.day edclv band new york ny selling day edclv band meet vegas hmu.got vape chamber empty need sealed eliquid day tia.anyone selling day ga looking friend.thought something maybe already know eh able call registration number bracelets verify registration valid remember years ago bought ticket somebody mean would suck find somebody reported stolen think would better find sooner later like get way security light red thought.looking one day pass
    

### Second step: Preprocessing

After getting the posts as text, we start with the preprocessing part.
For making the model faster and better, we will limit the vocabulary size.
Plus, we will use special tokens for unknown, start and end sentance.


```python
vocabulary_size = 600
unknown_token = "UNKNOWNTOKEN"
sentence_start_token = "SENTENCESTART"
sentence_end_token = "SENTENCEEND"
separator= "SEPARATOR"
```

And now we will replace the existing signs with the selected tokens:


```python
male_posts_as_text = male_posts_as_text.replace('\n', ' ')
male_posts_as_text = male_posts_as_text.replace('--',' '+ separator + ' ')
male_posts_as_text = male_posts_as_text.replace('.',' '+sentence_end_token +' '+ sentence_start_token+' ' )
```


```python
female_posts_as_text = female_posts_as_text.replace('\n', ' ')
female_posts_as_text = female_posts_as_text.replace('--',' '+ separator + ' ')
female_posts_as_text = female_posts_as_text.replace('.',' '+sentence_end_token +' '+ sentence_start_token+' ' )
```

And now we will split the text to list of words:


```python
male_posts_text = text_to_word_sequence(male_posts_as_text, lower=False, split=" ")
male_posts_text[0:10]
```




    ['many',
     'people',
     'got',
     'scammed',
     'weekend',
     'SENTENCEEND',
     'SENTENCESTART',
     'idk',
     'yall',
     'best']




```python
female_posts_text = text_to_word_sequence(female_posts_as_text, lower=False, split=" ")
female_posts_text[0:10]
```




    ['insomniac',
     'released',
     'statement',
     'man',
     'death',
     'SENTENCEEND',
     'SENTENCESTART',
     'know',
     'kaskade',
     'performed']



### third step: Building the models

After organizing our data as list of words and as text, we can build now our models.
Lets create a tolenizer:


```python
token_male = Tokenizer(num_words=600,char_level=False)
token_male.fit_on_texts(male_posts_text)
male_mtx = token_male.texts_to_matrix(male_posts_text, mode='binary')
male_mtx.shape
```




    (8357, 600)




```python
token_female = Tokenizer(num_words=600,char_level=False)
token_female.fit_on_texts(female_posts_text)
female_mtx = token_female.texts_to_matrix(female_posts_text, mode='binary')
female_mtx.shape
```




    (6343, 600)



And now lets build the models:


```python
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Flatten

input_male = male_mtx[:-1]
output_male = male_mtx[1:]

model_male = Sequential()
model_male.add(Embedding(input_dim=input_male.shape[1],output_dim= 42, input_length=input_male.shape[1]))
model_male.add(Flatten())
model_male.add(Dense(output_male.shape[1], activation='sigmoid'))
model_male.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
model_male.fit(input_male, y=output_male, batch_size=300, nb_epoch=50, verbose=1, validation_split=0.2)
```

    C:\Users\t-avadir\AppData\Local\Continuum\Anaconda3\lib\site-packages\keras\models.py:846: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      warnings.warn('The `nb_epoch` argument in `fit` '
    

    Train on 6684 samples, validate on 1672 samples
    Epoch 1/50
    6684/6684 [==============================] - 25s - loss: 4.9386 - acc: 0.0147 - val_loss: 4.5523 - val_acc: 0.0090
    Epoch 2/50
    6684/6684 [==============================] - 23s - loss: 4.6916 - acc: 0.0829 - val_loss: 4.4812 - val_acc: 0.1459
    Epoch 3/50
    6684/6684 [==============================] - 25s - loss: 4.5867 - acc: 0.1595 - val_loss: 4.5011 - val_acc: 0.0730
    Epoch 4/50
    6684/6684 [==============================] - 23s - loss: 4.5209 - acc: 0.0797 - val_loss: 4.3963 - val_acc: 0.0730
    Epoch 5/50
    6684/6684 [==============================] - 24s - loss: 4.4653 - acc: 0.0797 - val_loss: 4.4292 - val_acc: 0.0730
    Epoch 6/50
    6684/6684 [==============================] - 24s - loss: 4.3892 - acc: 0.0797 - val_loss: 4.3942 - val_acc: 0.0730
    Epoch 7/50
    6684/6684 [==============================] - 24s - loss: 4.2985 - acc: 0.1275 - val_loss: 4.3167 - val_acc: 0.1519
    Epoch 8/50
    6684/6684 [==============================] - 24s - loss: 4.1761 - acc: 0.1646 - val_loss: 4.2614 - val_acc: 0.1609
    Epoch 9/50
    6684/6684 [==============================] - 24s - loss: 4.0573 - acc: 0.1715 - val_loss: 4.2168 - val_acc: 0.1603
    Epoch 10/50
    6684/6684 [==============================] - 24s - loss: 3.9476 - acc: 0.1758 - val_loss: 4.1536 - val_acc: 0.1657
    Epoch 11/50
    6684/6684 [==============================] - 24s - loss: 3.8261 - acc: 0.1831 - val_loss: 4.1104 - val_acc: 0.1675
    Epoch 12/50
    6684/6684 [==============================] - 23s - loss: 3.7208 - acc: 0.1893 - val_loss: 4.0913 - val_acc: 0.1681
    Epoch 13/50
    6684/6684 [==============================] - 23s - loss: 3.6114 - acc: 0.1994 - val_loss: 4.0082 - val_acc: 0.1728
    Epoch 14/50
    6684/6684 [==============================] - 22s - loss: 3.5085 - acc: 0.2120 - val_loss: 4.0168 - val_acc: 0.1734
    Epoch 15/50
    6684/6684 [==============================] - 23s - loss: 3.4020 - acc: 0.2190 - val_loss: 3.9684 - val_acc: 0.1734
    Epoch 16/50
    6684/6684 [==============================] - 23s - loss: 3.3038 - acc: 0.2255 - val_loss: 4.0014 - val_acc: 0.1675
    Epoch 17/50
    6684/6684 [==============================] - 24s - loss: 3.2094 - acc: 0.2341 - val_loss: 3.9843 - val_acc: 0.1699
    Epoch 18/50
    6684/6684 [==============================] - 23s - loss: 3.1118 - acc: 0.2439 - val_loss: 3.9261 - val_acc: 0.1794
    Epoch 19/50
    6684/6684 [==============================] - 23s - loss: 3.0215 - acc: 0.2516 - val_loss: 3.9257 - val_acc: 0.1764
    Epoch 20/50
    6684/6684 [==============================] - 23s - loss: 2.9336 - acc: 0.2581 - val_loss: 3.9224 - val_acc: 0.1776
    Epoch 21/50
    6684/6684 [==============================] - 23s - loss: 2.8490 - acc: 0.2647 - val_loss: 3.9745 - val_acc: 0.1734
    Epoch 22/50
    6684/6684 [==============================] - 25s - loss: 2.7689 - acc: 0.2741 - val_loss: 3.9934 - val_acc: 0.1693
    Epoch 23/50
    6684/6684 [==============================] - 22s - loss: 2.6930 - acc: 0.2787 - val_loss: 3.9926 - val_acc: 0.1770
    Epoch 24/50
    6684/6684 [==============================] - 24s - loss: 2.6293 - acc: 0.2838 - val_loss: 4.0179 - val_acc: 0.1699
    Epoch 25/50
    6684/6684 [==============================] - 25s - loss: 2.5612 - acc: 0.2870 - val_loss: 4.0019 - val_acc: 0.1752
    Epoch 26/50
    6684/6684 [==============================] - 24s - loss: 2.5021 - acc: 0.2893 - val_loss: 4.0282 - val_acc: 0.1711
    Epoch 27/50
    6684/6684 [==============================] - 24s - loss: 2.4479 - acc: 0.2967 - val_loss: 4.0913 - val_acc: 0.1717
    Epoch 28/50
    6684/6684 [==============================] - 23s - loss: 2.3951 - acc: 0.3012 - val_loss: 4.1710 - val_acc: 0.1800
    Epoch 29/50
    6684/6684 [==============================] - 24s - loss: 2.3505 - acc: 0.3031 - val_loss: 4.1624 - val_acc: 0.1746
    Epoch 30/50
    6684/6684 [==============================] - 25s - loss: 2.3112 - acc: 0.3066 - val_loss: 4.1757 - val_acc: 0.1818
    Epoch 31/50
    6684/6684 [==============================] - 26s - loss: 2.2691 - acc: 0.3079 - val_loss: 4.1915 - val_acc: 0.1776
    Epoch 32/50
    6684/6684 [==============================] - 24s - loss: 2.2406 - acc: 0.3066 - val_loss: 4.1984 - val_acc: 0.1687
    Epoch 33/50
    6684/6684 [==============================] - 24s - loss: 2.2034 - acc: 0.3085 - val_loss: 4.2753 - val_acc: 0.1746
    Epoch 34/50
    6684/6684 [==============================] - 24s - loss: 2.1772 - acc: 0.3100 - val_loss: 4.4247 - val_acc: 0.1830
    Epoch 35/50
    6684/6684 [==============================] - 24s - loss: 2.1565 - acc: 0.3128 - val_loss: 4.2930 - val_acc: 0.1717
    Epoch 36/50
    6684/6684 [==============================] - 23s - loss: 2.1343 - acc: 0.3113 - val_loss: 4.3345 - val_acc: 0.1711
    Epoch 37/50
    6684/6684 [==============================] - 24s - loss: 2.1137 - acc: 0.3109 - val_loss: 4.5122 - val_acc: 0.1794
    Epoch 38/50
    6684/6684 [==============================] - 23s - loss: 2.0960 - acc: 0.3128 - val_loss: 4.5301 - val_acc: 0.1866
    Epoch 39/50
    6684/6684 [==============================] - 24s - loss: 2.0847 - acc: 0.3077 - val_loss: 4.5612 - val_acc: 0.1812
    Epoch 40/50
    6684/6684 [==============================] - 23s - loss: 2.0703 - acc: 0.3089 - val_loss: 4.5472 - val_acc: 0.1812
    Epoch 41/50
    6684/6684 [==============================] - 23s - loss: 2.0674 - acc: 0.3083 - val_loss: 4.5152 - val_acc: 0.1758
    Epoch 42/50
    6684/6684 [==============================] - 22s - loss: 2.0482 - acc: 0.3100 - val_loss: 4.5885 - val_acc: 0.1776
    Epoch 43/50
    6684/6684 [==============================] - 23s - loss: 2.0526 - acc: 0.3121 - val_loss: 4.5818 - val_acc: 0.1699
    Epoch 44/50
    6684/6684 [==============================] - 23s - loss: 2.0369 - acc: 0.3082 - val_loss: 4.7840 - val_acc: 0.1818
    Epoch 45/50
    6684/6684 [==============================] - 23s - loss: 2.0348 - acc: 0.3046 - val_loss: 4.7491 - val_acc: 0.1848
    Epoch 46/50
    6684/6684 [==============================] - 22s - loss: 2.0243 - acc: 0.3085 - val_loss: 4.7502 - val_acc: 0.1824
    Epoch 47/50
    6684/6684 [==============================] - 23s - loss: 2.0231 - acc: 0.3054 - val_loss: 4.6468 - val_acc: 0.1717
    Epoch 48/50
    6684/6684 [==============================] - 23s - loss: 2.0163 - acc: 0.3054 - val_loss: 4.6601 - val_acc: 0.1764
    Epoch 49/50
    6684/6684 [==============================] - 25s - loss: 2.0190 - acc: 0.3070 - val_loss: 4.7154 - val_acc: 0.1717
    Epoch 50/50
    6684/6684 [==============================] - 23s - loss: 2.0161 - acc: 0.3063 - val_loss: 4.6953 - val_acc: 0.1705
    




    <keras.callbacks.History at 0x15c0903f208>




```python
input_female = female_mtx[:-1]
output_female = female_mtx[1:]

model_female = Sequential()
model_female.add(Embedding(input_dim=input_female.shape[1],output_dim= 42, input_length=input_female.shape[1]))
model_female.add(Flatten())
model_female.add(Dense(output_female.shape[1], activation='sigmoid'))
model_female.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])
model_female.fit(input_female, y=output_female, batch_size=300, nb_epoch=50, verbose=1, validation_split=0.2)
```

    C:\Users\t-avadir\AppData\Local\Continuum\Anaconda3\lib\site-packages\keras\models.py:846: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      warnings.warn('The `nb_epoch` argument in `fit` '
    

    Train on 5073 samples, validate on 1269 samples
    Epoch 1/50
    5073/5073 [==============================] - 18s - loss: 4.9271 - acc: 0.0140 - val_loss: 4.7283 - val_acc: 0.0102
    Epoch 2/50
    5073/5073 [==============================] - 17s - loss: 4.6522 - acc: 0.0686 - val_loss: 4.7638 - val_acc: 0.0977
    Epoch 3/50
    5073/5073 [==============================] - 19s - loss: 4.5788 - acc: 0.1335 - val_loss: 4.7817 - val_acc: 0.0977
    Epoch 4/50
    5073/5073 [==============================] - 18s - loss: 4.5188 - acc: 0.1100 - val_loss: 4.7726 - val_acc: 0.0977
    Epoch 5/50
    5073/5073 [==============================] - 18s - loss: 4.4745 - acc: 0.1151 - val_loss: 4.8227 - val_acc: 0.0489
    Epoch 6/50
    5073/5073 [==============================] - 17s - loss: 4.4227 - acc: 0.0735 - val_loss: 4.7622 - val_acc: 0.0489
    Epoch 7/50
    5073/5073 [==============================] - 17s - loss: 4.3744 - acc: 0.0735 - val_loss: 4.8014 - val_acc: 0.0489
    Epoch 8/50
    5073/5073 [==============================] - 18s - loss: 4.3146 - acc: 0.0735 - val_loss: 4.7943 - val_acc: 0.0489
    Epoch 9/50
    5073/5073 [==============================] - 18s - loss: 4.2277 - acc: 0.1301 - val_loss: 4.8480 - val_acc: 0.0977
    Epoch 10/50
    5073/5073 [==============================] - 18s - loss: 4.1424 - acc: 0.1543 - val_loss: 4.7887 - val_acc: 0.0977
    Epoch 11/50
    5073/5073 [==============================] - 19s - loss: 4.0473 - acc: 0.1595 - val_loss: 4.7478 - val_acc: 0.1190
    Epoch 12/50
    5073/5073 [==============================] - 18s - loss: 3.9544 - acc: 0.1727 - val_loss: 4.7095 - val_acc: 0.1103
    Epoch 13/50
    5073/5073 [==============================] - 17s - loss: 3.8683 - acc: 0.1798 - val_loss: 4.7129 - val_acc: 0.1229
    Epoch 14/50
    5073/5073 [==============================] - 17s - loss: 3.7876 - acc: 0.1829 - val_loss: 4.6759 - val_acc: 0.1166
    Epoch 15/50
    5073/5073 [==============================] - 17s - loss: 3.7056 - acc: 0.1881 - val_loss: 4.6428 - val_acc: 0.1182
    Epoch 16/50
    5073/5073 [==============================] - 17s - loss: 3.6213 - acc: 0.1928 - val_loss: 4.5764 - val_acc: 0.1253
    Epoch 17/50
    5073/5073 [==============================] - 17s - loss: 3.5431 - acc: 0.1963 - val_loss: 4.5383 - val_acc: 0.1245
    Epoch 18/50
    5073/5073 [==============================] - 17s - loss: 3.4701 - acc: 0.2015 - val_loss: 4.5325 - val_acc: 0.1229
    Epoch 19/50
    5073/5073 [==============================] - 17s - loss: 3.3949 - acc: 0.2066 - val_loss: 4.5245 - val_acc: 0.1237
    Epoch 20/50
    5073/5073 [==============================] - 17s - loss: 3.3226 - acc: 0.2125 - val_loss: 4.5822 - val_acc: 0.1221
    Epoch 21/50
    5073/5073 [==============================] - 17s - loss: 3.2498 - acc: 0.2123 - val_loss: 4.5371 - val_acc: 0.1245
    Epoch 22/50
    5073/5073 [==============================] - 17s - loss: 3.1838 - acc: 0.2198 - val_loss: 4.5126 - val_acc: 0.1292
    Epoch 23/50
    5073/5073 [==============================] - 18s - loss: 3.1097 - acc: 0.2279 - val_loss: 4.5285 - val_acc: 0.1269
    Epoch 24/50
    5073/5073 [==============================] - 17s - loss: 3.0465 - acc: 0.2342 - val_loss: 4.5158 - val_acc: 0.1284
    Epoch 25/50
    5073/5073 [==============================] - 17s - loss: 2.9784 - acc: 0.2397 - val_loss: 4.4961 - val_acc: 0.1300
    Epoch 26/50
    5073/5073 [==============================] - 18s - loss: 2.9186 - acc: 0.2454 - val_loss: 4.5286 - val_acc: 0.1308
    Epoch 27/50
    5073/5073 [==============================] - 17s - loss: 2.8577 - acc: 0.2529 - val_loss: 4.4812 - val_acc: 0.1348
    Epoch 28/50
    5073/5073 [==============================] - 17s - loss: 2.7939 - acc: 0.2616 - val_loss: 4.4996 - val_acc: 0.1348
    Epoch 29/50
    5073/5073 [==============================] - 18s - loss: 2.7398 - acc: 0.2677 - val_loss: 4.5264 - val_acc: 0.1379
    Epoch 30/50
    5073/5073 [==============================] - 17s - loss: 2.6817 - acc: 0.2714 - val_loss: 4.5125 - val_acc: 0.1379
    Epoch 31/50
    5073/5073 [==============================] - 17s - loss: 2.6291 - acc: 0.2770 - val_loss: 4.5127 - val_acc: 0.1418
    Epoch 32/50
    5073/5073 [==============================] - 18s - loss: 2.5773 - acc: 0.2805 - val_loss: 4.5232 - val_acc: 0.1418
    Epoch 33/50
    5073/5073 [==============================] - 18s - loss: 2.5258 - acc: 0.2908 - val_loss: 4.5037 - val_acc: 0.1442
    Epoch 34/50
    5073/5073 [==============================] - 18s - loss: 2.4791 - acc: 0.2917 - val_loss: 4.6893 - val_acc: 0.1442
    Epoch 35/50
    5073/5073 [==============================] - 18s - loss: 2.4380 - acc: 0.2949 - val_loss: 4.4978 - val_acc: 0.1411
    Epoch 36/50
    5073/5073 [==============================] - 17s - loss: 2.3975 - acc: 0.2973 - val_loss: 4.6030 - val_acc: 0.1403
    Epoch 37/50
    5073/5073 [==============================] - 17s - loss: 2.3539 - acc: 0.2963 - val_loss: 4.5591 - val_acc: 0.1411
    Epoch 38/50
    5073/5073 [==============================] - 18s - loss: 2.3174 - acc: 0.3012 - val_loss: 4.6394 - val_acc: 0.1395
    Epoch 39/50
    5073/5073 [==============================] - 17s - loss: 2.2886 - acc: 0.3030 - val_loss: 4.7331 - val_acc: 0.1418
    Epoch 40/50
    5073/5073 [==============================] - 18s - loss: 2.2529 - acc: 0.3051 - val_loss: 4.6475 - val_acc: 0.1442
    Epoch 41/50
    5073/5073 [==============================] - 17s - loss: 2.2248 - acc: 0.3097 - val_loss: 4.7501 - val_acc: 0.1426
    Epoch 42/50
    5073/5073 [==============================] - 18s - loss: 2.1968 - acc: 0.3055 - val_loss: 4.7390 - val_acc: 0.1466
    Epoch 43/50
    5073/5073 [==============================] - 20s - loss: 2.1709 - acc: 0.3053 - val_loss: 4.8382 - val_acc: 0.1458
    Epoch 44/50
    5073/5073 [==============================] - 17s - loss: 2.1513 - acc: 0.3042 - val_loss: 4.7870 - val_acc: 0.1426
    Epoch 45/50
    5073/5073 [==============================] - 17s - loss: 2.1260 - acc: 0.3061 - val_loss: 4.8658 - val_acc: 0.1450
    Epoch 46/50
    5073/5073 [==============================] - 18s - loss: 2.1131 - acc: 0.3049 - val_loss: 4.8798 - val_acc: 0.1458
    Epoch 47/50
    5073/5073 [==============================] - 17s - loss: 2.0928 - acc: 0.3049 - val_loss: 4.8903 - val_acc: 0.1474
    Epoch 48/50
    5073/5073 [==============================] - 18s - loss: 2.0850 - acc: 0.3063 - val_loss: 4.9004 - val_acc: 0.1426
    Epoch 49/50
    5073/5073 [==============================] - 19s - loss: 2.0656 - acc: 0.3059 - val_loss: 5.0571 - val_acc: 0.1513
    Epoch 50/50
    5073/5073 [==============================] - 18s - loss: 2.0561 - acc: 0.3040 - val_loss: 4.9945 - val_acc: 0.1403
    




    <keras.callbacks.History at 0x15c09187940>



### 4th step: Generating sentences

This method is used for getting the next word of a given word.


```python
def get_next(text,token,model,fullmtx,fullText):
    tmp = text_to_word_sequence(text, lower=False, split=" ")
    tmp = token.texts_to_matrix(tmp, mode='binary')
    p = model.predict(tmp)
    bestMatches = p.argsort() [0][-20:]
    bestMatch = np.random.choice(bestMatches,1)[0]
    next_idx = np.min(np.where(fullmtx[:,bestMatch]>0))
    return fullText[next_idx]
```


```python
def generatePosts(text, token, text_mtx, model):
    posts=[]
    for i in range(0,100):
        word = sentence_start_token
        post=""
        while word != sentence_end_token:
            word = get_next(word, token, model, text_mtx, text)
            if word != sentence_end_token:
                post=post+' '+word
        posts.append(post)
    
    return posts
```


```python
male_generated_posts = generatePosts(male_posts_text, token_male, male_mtx, model_male)
male_generated_posts[:10]
```




    [' vip edc ticket obo good edc day tix good deals going vacation meet',
     ' selling kandi get thank tickets tonight pregame ga pass call anyone extra room meet around friday made free entry day vip sale low make selling beyond free merch ga ticket name meet strip friday meet cosmo pm reservations right ga passes village pass las got fire ga parking tonight sat las made need somewhere meet tonight better',
     ' ga days best rides tonight help',
     ' https people selling linq got vegas anywhere see today asking day many got day vip get em edc',
     ' edc town day passes san edc wristband sat edc tix got spot ga willing let tonight',
     ' day ga pass festival ga cash rides free get scammed tonight tomorrow morning get guest edc tickets two edc babes people tonight sell want thank day passes help',
     ' edc day ticket willing good deals tonight would las lol las vegas right hotel good las las people paying got cash good going sunday shoot las rides get parking day airport day parking good morning san tonight tomorrow meet',
     ' vip las made las rides las see day please let lol',
     ' anybody wants play tonight pregame see us las lol edc day tix get wristbands please help day vip ticket name ga cash get enjoy want coming ga tickets green see axwell edc sure got cancelled want thank guys whats app going idk good tickets',
     ' looking needing meet las got tree tonight need two']




```python
female_generated_posts = generatePosts(female_posts_text, token_female, female_mtx, model_female)
female_generated_posts[:10]
```




    [' hey going marshmello get money going wristband buy going attending time',
     ' selling ticket per vegas la pm let linq vegas painted',
     ' edc first night shuttle lines tickets yet please tell time going get festival grounds vegas la time meet inside tonight wants share pm need wristbands linq shuttle pass las wait vegas meet vegas within selling sunday edc wristband help need ticket downtown going las edc wristband could tickets scan vegas village especially',
     ' got day lets day also edc bracelets meet back tomorrow heading please thank amazing pm plz please pop vegas little edc remember free going day shuttle stop vegas painted need wristbands day general selling one ga last resort please edc remember tonight intrigue trying stay circus wants selling chainsmokers edc sat pm let shit vegas today edc wristband',
     ' day shuttle lines linq pm bus las u get uber one sunday located time shuttle pass willing take selling made selling ticket linq shuttle stop tickets sale hmu linq pop time single people need wristbands day sunday located shuttle sale hmu day day pass lmk get back going edc high get hot get us time need',
     ' need care need one wristband buy big red time life married free need linq edc day sunday santa pop many get back tomorrow message pop first linq one sale pm going get wristband plz free pop please vegas village lot near time year get existing edc ya pm need ticket san day called edc tickets tonight tomorrow heading one',
     ' two shuttle pop going attending day admission selling made anyone know kaskade one experience ya linq tonight edc first day ga saturday linq going las id pm edc pass pm interested vegas meet around going need wristbands need wristbands trying anyone interested pop vegas festival need trade day admission tickets vancouver wa wristband',
     ' sold shuttle last tonight saturday going marshmello please linq anyone going split tickets stolen meet back go anyone zedd',
     ' sunday',
     ' lost shuttle ticket please pm need ticket linq asking one please let pm get made edc day vegas earlier u going day shuttle pass message edc ga pass anyone chance going marshmello day general pm edc']



### 5th step: Write the results to a .csv file
Now, after creating our posts, let's convert them to DataFrame and write them to a file, called "generated_posts.csv".


```python
df_male = pd.DataFrame({'message':male_generated_posts})
df_male['gender'] = 'male'
df_male.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>message</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vip edc ticket obo good edc day tix good deal...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>selling kandi get thank tickets tonight prega...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ga days best rides tonight help</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https people selling linq got vegas anywhere ...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>4</th>
      <td>edc town day passes san edc wristband sat edc...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>day ga pass festival ga cash rides free get s...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>edc day ticket willing good deals tonight wou...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>7</th>
      <td>vip las made las rides las see day please let...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>8</th>
      <td>anybody wants play tonight pregame see us las...</td>
      <td>male</td>
    </tr>
    <tr>
      <th>9</th>
      <td>looking needing meet las got tree tonight nee...</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_female = pd.DataFrame({'message':female_generated_posts})
df_female['gender'] = 'female'
df_female.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>message</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hey going marshmello get money going wristban...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>selling ticket per vegas la pm let linq vegas...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>edc first night shuttle lines tickets yet ple...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>got day lets day also edc bracelets meet back...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>day shuttle lines linq pm bus las u get uber ...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>need care need one wristband buy big red time...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two shuttle pop going attending day admission...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sold shuttle last tonight saturday going mars...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sunday</td>
      <td>female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>lost shuttle ticket please pm need ticket lin...</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>




```python
frames = [df_female, df_male]
generated_posts = pd.concat(frames)

# Lets reorder the columns so they will fit to our previus files
cols = generated_posts.columns.tolist()
cols = cols[-1:] + cols[:-1]
generated_posts = generated_posts[cols] 

generated_posts.to_csv("generated_posts.csv")
```

# Part D
In this part we will check the posts we generated in part C, using the models from part B.

### First step: Reading the data

The generated posts are written in a file called: "generated_posts.csv".
We can read them from there, but actually we can use the var generated_posts.


```python
generated_posts.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>message</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>hey going marshmello get money going wristban...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>1</th>
      <td>selling ticket per vegas la pm let linq vegas...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>edc first night shuttle lines tickets yet ple...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>3</th>
      <td>got day lets day also edc bracelets meet back...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>day shuttle lines linq pm bus las u get uber ...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>5</th>
      <td>need care need one wristband buy big red time...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two shuttle pop going attending day admission...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sold shuttle last tonight saturday going mars...</td>
      <td>female</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sunday</td>
      <td>female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>lost shuttle ticket please pm need ticket lin...</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's do the same steps from part B:


```python
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 5000) 

clean_generated_post = generated_posts['message'].apply(lambda post: post_to_words(post))
posts_data_features2 = vectorizer.fit_transform(clean_generated_post)
posts_data_features2 = posts_data_features2.toarray()

#split to train & test
train_generated_posts = np.random.rand(len(clean_generated_post)) < 0.75
train_generated_message = posts_data_features2[train_generated_posts]
train_generated_gender = generated_posts.loc[train_generated_posts,"gender"]

test_generated_message = posts_data_features2[~train_generated_posts]
test_generated_gender = generated_posts.loc[~train_generated_posts,"gender"]
```

### Second step: Checking our posts using several models:

### First model:  K-Neighbors model.


```python
KNeighbors = KNeighborsClassifier(n_neighbors=130) 

KNeighbors = KNeighbors.fit( train_generated_message, train_generated_gender )

score = KNeighbors.score(test_generated_message, test_generated_gender )
score

```




    0.76000000000000001



### Second model: Gradient Boosting model.


```python
GradientBoosting = GradientBoostingClassifier( n_estimators = 45 ) 

GradientBoosting = GradientBoosting.fit( train_generated_message, train_generated_gender )

score = GradientBoosting.score(test_generated_message, test_generated_gender )
score

```




    0.83999999999999997



### Third model: Decision Tree model.


```python
DecisionTree= DecisionTreeClassifier(random_state = 1) 

DecisionTree = DecisionTree.fit( train_generated_message, train_generated_gender )

score = DecisionTree.score(test_generated_message,test_generated_gender )
score
```




    0.76000000000000001



### 4th model: Logistic Regression model.


```python
from sklearn.linear_model import LogisticRegression

LogisticRegression= LogisticRegression() 

LogisticRegression = LogisticRegression.fit( train_generated_message, train_generated_gender )

score = LogisticRegression.score(test_generated_message,test_generated_gender )
score
```




    0.90000000000000002



### 5th model: Random Forest model.


```python
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier= RandomForestClassifier( n_estimators = 80 ) 

RandomForestClassifier = RandomForestClassifier.fit( train_generated_message, train_generated_gender )

score = RandomForestClassifier.score(test_generated_message,test_generated_gender )
score
```




    0.92000000000000004



## Summarize

The best result that we got was with Random Forest model, with score of ~0.92.
We can see that our models, also the generation model and the classifiers, are works very well.
