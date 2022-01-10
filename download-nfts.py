import threading
import requests
from tqdm import tqdm
from pathlib import Path
from opensea import OpenseaAPI
from api_keys import API_KEY

LOCK = threading.Lock()
download_count = 0
threads_created = []
MAX_THREADS = 10
ACTIVE_THREADS = 0

api = OpenseaAPI(apikey=API_KEY)


def get_image_urls(slug, total=None, limit=30):
    global api
    count = 1
    col_stats = api.collection_stats(slug)

    print("")

    image_urls = []

    if "stats" in col_stats.keys():
        if "count" in col_stats["stats"].keys():

            nft_count = int(col_stats["stats"]["count"])
            if total is None:
                total = nft_count

            last = False

            while True:

                print(f"Getting URLs: {count - 1}/{total}", end="\r")

                if total - len(image_urls) < limit:
                    next_count = count + total - len(image_urls)
                else:
                    next_count = count + limit

                if next_count > nft_count:
                    next_count = nft_count + 1
                    last = True

                token_ids = list(range(count, next_count))

                if len(token_ids) > 0:

                    asset_list = api.assets(collection=slug, token_ids=token_ids)[
                        "assets"
                    ]

                    if len(asset_list) > 0:

                        for a in asset_list:
                            if "image_url" in a.keys():
                                image_urls.append(a["image_url"])

                count = next_count

                if last or (len(image_urls) >= total):
                    print("\nDone!")
                    break
            return image_urls

    return []


def check_threads():
    global ACTIVE_THREADS
    while ACTIVE_THREADS >= MAX_THREADS:
        pass


# DOWNLOAD ALL IMAGES FROM THAT URL
def download_images(images, folder_name, offset=0):
    global LOCK
    global download_count
    global ACTIVE_THREADS
    try:

        # initial count is zero
        count = 0

        # checking if images is not zero
        if len(images) != 0:
            for i, image_link in enumerate(images):

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
                        LOCK.acquire(True)
                        download_count += 1
                        LOCK.release()
                except:
                    pass

    except Exception as e:
        print(f"Failed to download images! Exception: {e}")

    finally:
        ACTIVE_THREADS -= 1


dataset_path = Path("./new_dataset/")

if not dataset_path.exists():
    dataset_path.mkdir()


url_list = [
    # "pop-wonder-editions",
    # "cryptocoven",
    # "clonex",
    # "boredapeyachtclub",
    # "cryptopunks",
    # "womenandweapons",
    # "chinese-zodiac-metaverse-by-yassartlabs",
    # "bofadeeznuts",
    # "divineanarchy",
    # "slotienft",
    # "doodles-official",
    # "smilesssvrs",
    "apocalyptic-apes",
    "earc",
    "ape-gang-old",
    "great-ape-society",
    "apes-in-space-nft",
    "apesofspace-official",
]

# print(len(get_image_urls("pop-wonder-editions", 10000)))
# exit()

t = tqdm(url_list)


for artist in t:

    artist_path = Path(f"./new_dataset/{artist}/")
    if not artist_path.exists():
        artist_path.mkdir()

    t.set_description(
        f"Artist: {artist} | Downloaded images: {download_count} | Active Threads: {ACTIVE_THREADS}"
    )

    total_images = get_image_urls(artist, 10000)

    check_threads()

    new_thread = threading.Thread(
        target=download_images,
        args=(total_images, f"new_dataset/{artist}"),
    )
    new_thread.start()
    threads_created.append(new_thread)
    ACTIVE_THREADS += 1


print("Waiting for downloads to finish!")
for thread in threads_created:
    thread.join()
