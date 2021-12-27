import threading
import time
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from pathlib import Path
from selenium.webdriver.firefox.options import Options

LOCK = threading.Lock()
download_count = 0
threads_created = []
MAX_THREADS = 10
ACTIVE_THREADS = 0


def check_threads():
    global ACTIVE_THREADS
    while ACTIVE_THREADS >= 10:
        pass


# DOWNLOAD ALL IMAGES FROM THAT URL
def download_images(images, folder_name, offset=0):
    global LOCK
    global download_count
    global ACTIVE_THREADS
    LOCK.acquire(True)

    try:

        # initial count is zero
        count = 0

        # checking if images is not zero
        if len(images) != 0:
            for i, image in enumerate(images):
                # From image tag ,Fetch image Source URL

                # 1.data-srcset
                # 2.data-src
                # 3.data-fallback-src
                # 4.src

                # Here we will use exception handling

                # first we will search for "data-srcset" in img tag
                try:
                    # In image tag ,searching for "data-srcset"
                    image_link = image["data-srcset"]

                # then we will search for "data-src" in img
                # tag and so on..
                except:
                    try:
                        # In image tag ,searching for "data-src"
                        image_link = image["data-src"]
                    except:
                        try:
                            # In image tag ,searching for "data-fallback-src"
                            image_link = image["data-fallback-src"]
                        except:
                            try:
                                # In image tag ,searching for "src"
                                image_link = image["src"]

                            # if no Source URL found
                            except:
                                pass

                # After getting Image Source URL
                # We will try to get the content of image
                try:
                    r = requests.get(image_link).content
                    try:

                        # possibility of decode
                        r = str(r, "utf-8")

                    except UnicodeDecodeError:

                        # After checking above condition, Image Download start
                        with open(f"{folder_name}/images{i+1+offset}.jpg", "wb+") as f:
                            f.write(r)

                        # counting number of image downloaded
                        count += 1
                except:
                    pass

            # There might be possible, that all
            # images not download
            # if all images download
            download_count += count
        else:
            download_count += 0

    except Exception as e:
        print(f"Failed to download images! Exception: {e}")

    finally:
        ACTIVE_THREADS -= 1
        LOCK.release()


dataset_path = Path("./new_dataset/")

if not dataset_path.exists():
    dataset_path.mkdir()


base_url = "https://opensea.io/assets/"

query_desc = "?search[resultModel]=ASSETS&search[sortAscending]=false&search[sortBy]=LAST_SALE_PRICE"
query_asce = "?search[resultModel]=ASSETS&search[sortAscending]=true&search[sortBy]=LAST_SALE_PRICE"

url_list = [
    # "pop-wonder-editions",
    # "cryptocoven",
    "clonex",
    "boredapeyachtclub",
    "cryptopunks",
    "womenandweapons",
    "chinese-zodiac-metaverse-by-yassartlabs",
    "bofadeeznuts",
    "divineanarchy",
    "slotienft",
    "doodles-official",
    "smilesssvrs",
]

t = tqdm(url_list)

options = Options()
options.headless = True
options.set_preference("browser.cache.disk.enable", False)
options.set_preference("browser.cache.memory.enable", False)
options.set_preference("browser.cache.offline.enable", False)
options.set_preference("network.http.use-cache", False)

for artist in t:
    ##### Web scrapper for infinite scrolling page #####
    artist_path = Path(f"./new_dataset/{artist}/")
    if not artist_path.exists():
        artist_path.mkdir()

    t.set_description(
        f"Artist: {artist} | Downloaded images: {download_count} | Active Threads: {ACTIVE_THREADS}"
    )
    driver = webdriver.Firefox(options=options)
    driver.get(base_url + artist + query_desc)
    time.sleep(2)  # Allow 2 seconds for the web page to open
    scroll_pause_time = 1
    driver.find_element_by_class_name("bnWGYU").click()
    driver.find_element_by_class_name("SearchFilter--header-button-container").click()
    screen_height = driver.execute_script("return window.screen.height;")
    i = 1

    total_images = set()
    saved_offset = 0
    while True:
        try:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            images = set(soup.findAll("img"))
            total_images = set.union(total_images, images)
            print(f"└─── Images found: {len(total_images) + saved_offset}", end="\r")
            if len(total_images) >= 5100:
                print(
                    f"└─── Images exceeded 5100! | Downloading {len(total_images)} images..."
                )
                check_threads()
                new_thread = threading.Thread(
                    target=download_images,
                    args=(list(total_images), f"new_dataset/{artist}"),
                )
                new_thread.start()
                threads_created.append(new_thread)
                ACTIVE_THREADS += 1
                saved_offset = len(total_images) + 1
                total_images = set()
                driver.quit()
                del driver
                driver = webdriver.Firefox(options=options)
                driver.get(base_url + artist + query_asce)
                time.sleep(2)  # Allow 2 seconds for the web page to open
                driver.find_element_by_class_name("bnWGYU").click()
                driver.find_element_by_class_name(
                    "SearchFilter--header-button-container"
                ).click()
                screen_height = driver.execute_script("return window.screen.height;")
                i = 1
            # scroll one screen height each time
            driver.execute_script(f"window.scrollTo(0, {screen_height}*{i});")
            i += 1
            time.sleep(scroll_pause_time)
            # update scroll height each time after scrolled, as the scroll height can change after we scrolled the page
            scroll_height = driver.execute_script("return document.body.scrollHeight;")
            del soup
            if len(total_images) + saved_offset >= 10001:
                break
            # Break the loop when the height we need to scroll to is larger than the total scroll height
            if (screen_height) * i > scroll_height:
                break
        except KeyboardInterrupt:
            break

    driver.quit()
    del driver

    check_threads()

    new_thread = threading.Thread(
        target=download_images,
        args=(list(total_images), f"new_dataset/{artist}", saved_offset),
    )
    new_thread.start()
    threads_created.append(new_thread)
    ACTIVE_THREADS += 1


print("Waiting for downloads to finish!")
for thread in threads_created:
    thread.join()
