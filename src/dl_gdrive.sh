
FILE_ID=1io-pusybNWefz4ueIPxkA8ErQ3KQaljD
TOKEN=ya29.a0ARrdaM_JrF9wtXSk-XvY49IWorG7pPyzBrVOdnsErU5_X_eZ1-M9cPrwVa650pArwGNFph5gUmh4fsHxCC5wDJfPlPqIh1cUWv7YSl1tlTbdjyQ3hfPLs6E47kCdKVeVNTHtKV9ppkmvi7iwJxMcKaGOD6aX
curl -H "Authorization: Bearer $TOKEN" https://www.googleapis.com/drive/v3/files/$FILE_ID?alt=media -o arxivL-intro.tar
