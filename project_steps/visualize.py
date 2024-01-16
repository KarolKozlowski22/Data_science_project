import matplotlib.pyplot as plt
import os 

current_dir = os.getcwd() + '/project_steps'
os.makedirs(f'{current_dir}/images', exist_ok=True)

def visualize_2d(train_images_2d, train_labels, method):
    plt.scatter(train_images_2d[:, 0], train_images_2d[:, 1], c=train_labels, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Fashion MNIST - {method} Reduced')
    plt.savefig(f'{current_dir}/images/{method}_2d.png')
    plt.close()
  

def visualize_3d(train_images_3d, train_labels, method):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_images_3d[:, 0], train_images_3d[:, 1], train_images_3d[:, 2], c=train_labels, cmap='viridis')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.title(f'Fashion MNIST - {method} Reduced')
    plt.savefig(f'{current_dir}/images/{method}_3d.png')
    plt.close()
   
