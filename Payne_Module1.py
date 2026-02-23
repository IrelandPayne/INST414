import json
import pandas as pd
import re
import matplotlib.pyplot as plt

#reading in data
with open("Amazon_popular_books_dataset.json", "r") as f:
    data = json.load(f)

rows = []

#Data cleaning ---

#getting numeric value for rating
for book in data:
    rating = None
    if book.get("rating") is not None:
        rating = float(str(book["rating"]).split()[0])

#getting main category
    main_genre = None
    if book.get("categories"):
        valid = [c for c in book["categories"] if str(c).lower() != "books"]
        if valid:
            main_genre = valid[0]

#cleaning author, ensuring only author is included
    cleaned_author = None
    author = book.get("brand")
    if author:
        author = author.replace("by ", "")
        author = re.sub(r"\(.*?\)", "", author)
        author = author.split(",")[0]
        author = author.split(";")[0]
        author = re.sub(r"\b(M\.?D\.?|Ph\.?D\.?)\b\.?", "", author, flags=re.I)
        author = " ".join(author.split())
        author = re.sub(r"\b([A-Z])\.\s+([A-Z])\.", r"\1.\2.", author)
        cleaned_author = author.strip()

#getting title for df
    title = book.get("title")

#df for further analysis
    rows.append({
        "Author": cleaned_author,
        "Title": title,
        "Genre": main_genre,
        "Rating": rating,
    })

#df creation
df = pd.DataFrame(rows)
#cleaning df, ensuring no duplicates and consistent formatting
df["Title"] = df["Title"].astype(str).str.strip().str.lower()
df = df.drop_duplicates(subset=["Title", "Author"]).reset_index(drop=True)

#getting top 10 genre and author
common_genres = df["Genre"].value_counts().head(10)
common_authors = df["Author"].value_counts().head(10)

#getting highest rating/highest rated genre and author
highest_rated = df[df["Rating"] == df["Rating"].max()]
best_genres = highest_rated["Genre"].value_counts().head(10)
best_authors = highest_rated["Author"].value_counts().head(10)

#printing outputs
#most common genres on best-selling list
print("\nTop 10 Genres:\n", common_genres)

#most common authors on best-selling list
print("\nTop 10 Authors:\n", common_authors)

#Genres with 4.9 rating (highest rating given)
print("\nHighest Rated Popular Genres:\n", best_genres)

#Authors with 4.9 rating (highest rating given)
print("\nHighest Rated Popular Authors:\n", best_authors)

#Graphs ---

#most common genre on best-selling list graph
plt.figure(figsize=(10, 5))
common_genres.sort_values().plot(kind="barh")
plt.title("Top 10 Authors Appearing in Amazon Best-Selling Books (10k+ Reviews)")
plt.xlabel("Number of Books")
plt.ylabel("Author")
plt.tight_layout()
plt.show()
plt.close() 

#most common authors on best-selling list graph
plt.figure(figsize=(10, 5))
common_authors.sort_values().plot(kind="barh")
plt.title("Top 10 Authors Appearing in Amazon Best-Selling Books (10k+ Reviews)")
plt.xlabel("Number of Books")
plt.ylabel("Author")
plt.tight_layout()
plt.show()
plt.close() 

#distribution of ratings on best-selling list
plt.figure(figsize=(8, 5))
df["Rating"].plot(kind="hist", bins=20)
plt.title("Distribution of Ratings in the Best-Selling Books Dataset")
plt.xlabel("Rating")
plt.ylabel("Number of Books")
plt.tight_layout()
plt.show()
plt.close() 

#distribution of genres with 4.9 rating (highest rating given)
plt.figure(figsize=(10, 5))
best_genres.sort_values().plot(kind="barh")
plt.title("Genres Among Highest-Rated Book(s) in Dataset")
plt.xlabel("Number of Highest-Rated Books")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()
plt.close() 

#distribution of authors with 4.9 rating (highest rating given)
plt.figure(figsize=(10, 5))
best_authors.sort_values().plot(kind="barh")
plt.title("Authors of Highest-Rated Book(s) in Dataset")
plt.xlabel("Number of Highest-Rated Books")
plt.ylabel("Author")
plt.tight_layout()
plt.show()
plt.close() 