# background + key messages unfiltered concatenation files:
token_sizes = {
    "Amphibian conservation" : 38929,
    "Bat Conservation": 61485,
    "Bee Conservation": 10995,
    "Biodiversity of Marine Artificial Structures": 67590,
    "Bird Conservation": 102677,
    "Butterfly and Moth Conservation": 92964,
    "Control of Freshwater Invasive Species": 57767,
    "Cool Farm Biodiversity": 8500,
    "Farmland Conservation": 38549,
    "Forest Conservation": 20765,
    "Grassland Conservation": 17872,
    "Management of Captive Animals": 19506,
    "Marine and Freshwater Mammal Conservation": 64402,
    "Marine Fish Conservation": 67111,
    "Marsh and Swamp Conservation": 280580,
    "Mediterranean Farmland": 32996,
    "Natural Pest Control": 9134,
    "Peatland Conservation": 66533,
    "Primate Conservation": 56608,
    "Reptile Conservation": 132607,
    "Shrubland and Heathland Conservation": 24084,
    "Soil Fertility": 9969,
    "Subtidal Benthic Invertebrate Conservation": 139508,
    "Sustainable Aquaculture": 3853,
    "Terrestrial Mammal Conservation": 104530
}
    
total = sum([v for v in token_sizes.values()])
mean = total / len(token_sizes)
print(mean)