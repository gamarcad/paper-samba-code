############### Jester large and small
nb_users = 24983
nb_jokes = 100
nb_jokes_small = 10
threshold_large = 4.5
threshold_small = 3.5
aggregate_per_joke_large = {i : 0. for i in range(1, nb_jokes+1)}
aggregate_per_joke_small = {i : 0. for i in [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]}

DATA_DIRECTORY="raw_data"


with open(f"{DATA_DIRECTORY}/jester.csv", "r") as f:
	line = f.readline()
	while line:
		line = list(map(lambda x: float(x) if x!='99' else -10, line.strip().split(",")))
		for i in range (1, nb_jokes+1):
			if line[i] >= threshold_large:
				aggregate_per_joke_large[i] += 1
			if i in aggregate_per_joke_small.keys() and line[i] >= threshold_small:
				aggregate_per_joke_small[i] += 1
		line = f.readline()

with open('JesterLarge.txt', 'w') as f:
	f.write(str(nb_jokes) + "\n")
	for i in aggregate_per_joke_large.keys():
		aggregate_per_joke_large[i] /= nb_users
		f.write(str(aggregate_per_joke_large[i]) + "\n")

with open('JesterSmall.txt', 'w') as f:
	f.write(str(nb_jokes_small) + "\n")
	for i in aggregate_per_joke_small.keys():
		aggregate_per_joke_small[i] /= nb_users
		f.write(str(aggregate_per_joke_small[i]) + "\n")


############### MovieLens
nb_users = 943
nb_movies = 100
threshold = 4
aggregate_per_movie = {i : 0. for i in range(1, nb_movies+1)}

with open(f"{DATA_DIRECTORY}/movie-lens.data", "r") as f: # This is a tab separated list of user id | item id | rating | timestamp
	line = f.readline()
	while line:
		line = list(map(lambda x: int(x), line.strip().split()[0:3]))
		if line[1] <= nb_movies and line[2] >= threshold: # we take only the first nb_movies movies, if rating above threshold
			aggregate_per_movie[line[1]] += 1
		line = f.readline()

with open('MovieLens.txt', "w") as f:
	f.write(str(nb_movies) + "\n")
	for i in aggregate_per_movie.keys():
		aggregate_per_movie[i] /= nb_users
		f.write(str(aggregate_per_movie[i]) + "\n")


