import requests


url = "https://thispersondoesnotexist.com/image"
urlStr = input("Enter a URL: ")

if urlStr != "":
    url = urlStr

numStr = input("Enter number of images: ")
nums = int(numStr)

for num in range(0, nums):
    r = requests.get(url, allow_redirects=True)
    fname = "image" + str(num + 1) + ".jpg"
    open(fname, "wb").write(r.content)
