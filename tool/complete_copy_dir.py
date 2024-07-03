#在训练之前（最好初始化时使用），加快训练速度
 # 开始调用图片，得到张量的图片和标签
        input_folder = './shoe_img'  # 要调整的图片所在的文件夹名称
        output_folder = './new112_shoe_img'  # 调整后的图片所放的文件夹名称
        # new_size = (56, 56)
        # 调整为112*112
        new_size = (112, 112)
        # 确保输出目录存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for root, dirs, files in os.walk(input_folder):
            for filename in files:
                # Check if the file is an image
                if filename.lower().endswith('.jpg'):
                    # 构建完整的文件路径
                    img_path = os.path.join(root, filename)
                    # Open the image and resize it
                    img = Image.open(img_path)
                    img = img.resize(new_size)
                    # 构建新的文件名和保存路径，保持原有目录结构
                    relative_path = os.path.relpath(root, input_folder)
                    output_subfolder = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)
                    new_file_path = os.path.join(output_subfolder, filename)
                    # Save the resized image to the corresponding output subfolder
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(new_file_path)