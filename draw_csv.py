import matplotlib.pyplot as plt
import csv

# Initialize lists for rounds (x_points) and objective function values (y_points)
x_points = []
y_points = []

# Open the CSV file in read mode
with open('CHANGE THE FILE NAME.csv', mode='r', newline='') as file:
    reader = csv.reader(file)
    # Iterate over each row in the CSV file
    i = 1
    for row in reader:
        y_points.append(float(row[0]))
        x_points.append(i)
        i += 1

# Create a new figure with a larger width to increase the distance between points on the X-axis
plt.figure(figsize=(12, 6))

# Plot points as a time series with smaller markers
plt.plot(x_points, y_points, color='blue', marker='o', markersize=4)

# Set plot title and labels
plt.title('Objective function over episode')
plt.xlabel('Episode number')
plt.ylabel('Objective function value')

# Show plot
plt.grid(True)
plt.show()
