# ANOTHER WAY OF NORMALISING

# another way of normalising
hybrid_image_normalised = (hybrid_image - hybrid_image.min()) / (hybrid_image.max() - hybrid_image.min())
cv2.imshow('hybrid_image_normalised', hybrid_image_normalised)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, you can scale the normalized hybrid image back to the range [0, 255] for visualization
# it seems this looks different
hybrid_image_visualized = (hybrid_image_normalised * 255).astype(np.uint8)
cv2.imwrite("hybrid_image_visualized.jpg", hybrid_image_visualized)