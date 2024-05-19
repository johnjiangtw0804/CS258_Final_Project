import matplotlib.pyplot as plt
import csv

# rounds
x_points = []
# objective function
y_points = []
# Open the CSV file in read mode
with open('objective_over_episode.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    # Iterate over each row in the CSV file
    i = 1
    for row in reader:
        y_points.append(float(row[0]))
        x_points.append(i)
        i += 1
# Plot points
plt.scatter(x_points, y_points, color='red')

# Set plot title and labels
plt.title('Objective function over episode')
plt.xlabel('episode number')
plt.ylabel('Objective function value')

# Show plot
plt.grid(True)
plt.show()