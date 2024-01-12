Files for Book Recommendation in idea file

Notes:
Currently the Book Rec System gives the option of allowing users to sort books through either genre or title:
  Genre: 
    1. The system can accept multiple genre inputs so long as they are written as ex."Genre1, Genre2" " or "Dark, Fantasy"
    2. Book Recommendations based on genre are decided driven by the rating
  Title:
    1. The systen can only generate suggestions based on one title at a time
    2. Book recommendations based on a title are driven by book descriptions
      a. sklearn.feature_extraction.text and TfidfVectorize from sklearn.neighbors uses Nearest Neighbor to idenify books 
      with descriptions as similar as possible to the description of the book title that is inputted.
Future Updates:
    1. Recommendations by author
      a. Book_rec.csv's author column is not optimal for this sort of query, an update will be implented soon that has a csv file that has a more optimal author column.
      Once the two csv files are merged,and the recommendation function is modified, users will be able to choose if they want to search by author, title, or genre.

  
