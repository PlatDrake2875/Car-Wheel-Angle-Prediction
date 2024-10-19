import json
import base64
import pandas as pd
from bagpy import bagreader
import glob
import os
from tqdm import tqdm


def read_bag_files(directory):
    return glob.glob(os.path.join(directory, '*.bag'))


def read_image_data(b, topics, output_dir):
    image_data = {}
    for topic in topics:
        image_topic = b.message_by_topic(topic)
        if image_topic:
            df = pd.read_csv(image_topic)
            images = []
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing images for {topic}"):
                if isinstance(row['data'], str):
                    row['data'] = eval(row['data'])
                images.append(base64.b64encode(row['data']).decode('utf-8'))
            image_data[topic] = images
        else:
            print(f"No data found for {topic} or file does not exist.")
    return image_data


def read_steering_data(b, steering_topic):
    steering_data = b.message_by_topic(steering_topic)
    if not steering_data:
        print("No steering data available.")
        return None
    df = pd.read_csv(steering_data)
    return df[['Time', 'steering_wheel_angle', 'steering_wheel_torque', 'speed']].to_dict(orient='records')


def process_bag_file(bag_file):
    print(f"Processing: {bag_file}")
    b = bagreader(bag_file)

    image_topics = [
        '/center_camera/image_color/compressed',
        '/left_camera/image_color/compressed',
        '/right_camera/image_color/compressed'
    ]
    steering_topic = '/vehicle/steering_report'

    bag_name = os.path.splitext(os.path.basename(bag_file))[0]
    output_dir = os.path.join(os.path.dirname(bag_file), bag_name)
    os.makedirs(output_dir, exist_ok=True)

    image_data = read_image_data(b, image_topics, output_dir)
    steering_data = read_steering_data(b, steering_topic)

    min_length = min(len(image_data[topic]) for topic in image_topics if topic in image_data)
    combined_data = {}
    for i in range(min_length):
        key = f"{bag_name}_{i}"
        combined_data[key] = {
            "images": {
                "left": image_data['/left_camera/image_color/compressed'][i],
                "center": image_data['/center_camera/image_color/compressed'][i],
                "right": image_data['/right_camera/image_color/compressed'][i]
            },
            "timestamp": steering_data[i]['Time'],
            "angle": steering_data[i]['steering_wheel_angle'],
            "torque": steering_data[i]['steering_wheel_torque'],
            "speed": steering_data[i]['speed']
        }

    json_filename = os.path.join(output_dir, f"{bag_name}_combined.json")
    with open(json_filename, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined data saved to: {json_filename}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    bag_files = read_bag_files(data_dir)
    for bag_file in bag_files:
        process_bag_file(bag_file)
    print("Data extraction complete.")


if __name__ == '__main__':
    main()
