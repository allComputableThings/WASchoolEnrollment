from geopy.geocoders import Nominatim
from geopy.distance import geodesic

geolocator = Nominatim(user_agent="WashingtonSchoolEnrollment")

# from geopy.geocoders import Nominatim
# geolocator = Nominatim(user_agent="WashintonSchoolEnrollment")
# location = geolocator.geocode("175 5th Avenue NYC")

addr1 = "17110 148 AVE NE, Woodinville WA"
addr2 = "14075 172 AV NE, Redmond, WA"

loc1 = geolocator.geocode(addr1)
loc2 = geolocator.geocode(addr2)

coords1 = (loc1.latitude, loc1.longitude)
coords2 = (loc2.latitude, loc2.longitude)

distance_miles = geodesic(coords1, coords2).miles
distance_km = geodesic(coords1, coords2).km

print(distance_miles, "miles")