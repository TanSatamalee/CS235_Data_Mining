import java.net.*;
import java.io.*;

/* Imports Jsoup for crawler. */
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;


public class Scraper {
	public static int DELAY = 7;
	public static void main(String[] args) {
	
		try {
			
			
			String[] categories = new String[4];
			categories[0] = "data mining";
			categories[1] = "databases";
			categories[2] = "machine learning";
			categories[3] = "artificial intelligence";

			for (String cur_category : categories) {
				
		        int numOfPages = 20;
		        
		        //create the output file
		        File file = new File(cur_category + "2.txt");
		        file.createNewFile();
		        FileWriter writer = new FileWriter(file); 
		       
		    
		        //initialize the string that will be written
		        String answer = "";

		        //now start crawling the all 'numOfPages' pages
		        for(int i = 1;i<=numOfPages;i++) {
		        	//Create the initial request to read the first page 
					//and get the number of total results
		        	String linkToScrape = "http://www.wikicfp.com/cfp/call?conference="+
		        				      URLEncoder.encode(cur_category, "UTF-8") +"&page=" + i;
		        	String content = getPageFromUrl(linkToScrape);	        	
		        	//parse or store the content of page 'i' here in 'content'

		        	//YOUR CODE GOES HERE

		        	//creates a jsoup object to crawl and finds the table containing data
		        	Document doc = Jsoup.parse(content);
		        	Elements tbody1 = doc.select("tbody").select("tbody");
		        	Element table = tbody1.get(5);

		        	//crawls through data row by row
		        	Elements data_rows = table.select("tr");
		        	int n = 0;
		        	int elem_count = 0;

		        	for (Element row : data_rows) {
		        		if (n == 0) {
		        			n += 1;
		        			continue;
		        		}
		        		for (Element d : row.select("td")) {
		        			//if (elem_count == 0 || elem_count == 1 || elem_count == 4) {
		        			answer += d.text() + "\t";
		        			
		        			//}
		        			elem_count += 1;
		        		}
		        		if (elem_count > 5) {
		        			elem_count = 0;
		        			answer += "\n";
		        		}
		        		n += 1;
		        	}
		        	//IMPORTANT! Do not change the following:
		        	Thread.sleep(DELAY*1000); //rate-limit the queries
		        }

		        writer.write(answer);
		        writer.flush();
	        	writer.close();
	        	
	        }
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

	}
	
	/**
	 * Given a string URL returns a string with the page contents
	 * Adapted from example in 
	 * http://docs.oracle.com/javase/tutorial/networking/urls/readingWriting.html
	 * @param link
	 * @return
	 * @throws IOException
	 */
	public static String getPageFromUrl(String link) throws IOException {
		URL thePage = new URL(link);
        URLConnection yc = thePage.openConnection();
        BufferedReader in = new BufferedReader(new InputStreamReader(
                                    yc.getInputStream()));
        String inputLine;
        String output = "";
        while ((inputLine = in.readLine()) != null) {
        	output += inputLine + "\n";
        }
        in.close();
		return output;
	}
	
}
