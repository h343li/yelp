Unsupervised Lexicon-Based Sentiment Topic Model (ULSTM)
========
The field of sentiment identification of a given text is of vital importance nowadays, as opinions expressed by others can
have significant influence on our daily decision-making process. From a business perspective, analyzing sentiments of
customer reviews can help them better cater customers needs. With the emergence of Microblogging platforms,
researchers in Natural Language Processing (NLP) have increased interest in the automatic detection of sentiment out of
mass texts.

Supervised sentiment analysis, or opinion mining, has achieved state-of-the-art performance using variations of
Long Short-Term Memory Model. However, it came to our attention that such approach has certain constraints. Firstly,
supervised learning requires labeled training data, which is not readily available in the field of sentiment analysis.
Secondly, peoples quantifications of the same sentiment are highly subjective. For instance, an ok sushi dish might
receive a four-star from one but a three-star from the next. Furthermore, it is not feasible to find a well-structured dataset
with standardized sentiments. Hence, an unsupervised analysis is superior in terms of sentiment detection since it
requires neither a true label attached to the text nor a universal scoring standard.

Inspired by Taboada et al. (2011)[6] and Hu and Liu (2004)[3], we present the Unsupervised Lexicon-Based
Sentiment Topic Model (ULSTM) using a self-established sentiment lexicon. The current application of the model
mainly concerns with reviews from the Yelp Open Dataset, which contains almost 6-million customer reviews. The
model assigns each review a corresponding sentiment score based on its semantic meaning and syntactic structure.
Then, it applies Latent Dirichlet Allocation (Blei et al., 2003) (LDA)[2] to selected businesses, aiming to extract global
topics as highlights and/or opportunities for improvement of the restaurants. Using these Yelp reviews as a starting
point, we hope to extend the model to all public tweets/texts on different social platforms.

![GitHub Logo](/h343li/yelp/blob/master/ULSTM.pdf)

<object data="https://github.com/h343li/yelp/blob/master/ULSTM.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/h343li/yelp/blob/master/ULSTM.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/h343li/yelp/blob/master/ULSTM.pdf">Download PDF</a>.</p>
    </embed>
</object>
