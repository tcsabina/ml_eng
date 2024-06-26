from io import BytesIO
import requests
import zipfile

class DataAcquisition:
    def __init__(self, url):
        """
        Initialize the DataAcquisition with a URL.

        :param url: A string representing the URL from which to download a file.
        """
        self.url = url

    def download_and_extract(self):
        """
        Download a file from the given URL and extract its contents.

        This method downloads a file from the specified URL and assumes it is a zip file.
        It then extracts the contents of the zip file into the current directory.
        """
        print("Downloading file...")
        response = requests.get(self.url)
        response.raise_for_status()
        print("Unzipping file...")
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        zip_file.extractall("data")
