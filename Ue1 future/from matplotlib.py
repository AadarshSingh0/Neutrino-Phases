import matplotlib.pyplot as plt

# Define the four points of the quadrilateral
x = [1, 4, 3, 0]  # x-coordinates of the points
y = [1, 1, 4, 3]  # y-coordinates of the points

# Create the plot
plt.figure(figsize=(6, 6))

# Plot the edges of the quadrilateral
plt.plot(x + [x[0]], y + [y[0]], 'k-', label='Edges')  # Adding the first point at the end to close the shape

# Fill the quadrilateral
plt.fill(x, y, color='lightblue', alpha=0.5, label='Filled Region')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Filled Quadrilateral')

# Display the plot
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio for correct representation
plt.show()
