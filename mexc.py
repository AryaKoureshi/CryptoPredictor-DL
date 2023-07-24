from mexc_sdk import Trade, Spot, Base, Market, UserData, Common
api_key = "mx0vglbndXqDHEAzUO"
apiSecret = "e5cded1181b441a68f50600399a9a2fd" 
spot = Spot(api_key=api_key, api_secret=apiSecret)
trade = Trade(api_key=api_key, api_secret=apiSecret)

print("ping: ", spot.ping())
print("time: ", spot.time())
spot.exchange_info()


