# CS235_Data_Mining

## Crawler and Hadoop
This project involves finding the location of conferences in data mining, machine learning, databases, and AI. This data is then cleaned and analyzed for visualization purposes. There are two parts for this project

### Part 1
Build a crawler to crawl WikiCFP for the conferences and location every year. From this data, we use OpenRefine to clean the data obtained. This part is detailed in `report1.pdf` with the crawler in `Scraper.java`, the resulting crawled data in the four `.txt` files indicating the conference type, and the cleaned data in `final_data.tsv`.

### Part 2
Use Hadoop to compute various statistics, and then use a visualization tool to create a heatmap of number of conferences for each city over time. This part is detailed in `report2.pdf` with the code in the `hadoop` folder (`.java` file is the hadoop file that is converted to `.jar` and the respective `_ans` file is the results of hadoop. The visualization portion uses Google Drive Fusion Table app to help visualize the heatmap after some post-processing after hadoop using python and OpenRefine.
