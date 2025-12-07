#!/usr/bin/env python3
"""
Example script showing how to use the face database features.

After processing a video, you can:
1. Inspect the database
2. Save/load the database
3. Extract face crops for visualization
4. Adjust clustering threshold
"""

from facefusion import state_manager
from facefusion.video_face_database import (
    get_database_summary,
    get_all_clusters,
    get_cluster,
    save_database,
    load_database,
    save_cluster_face_crops
)

# Example 1: Set clustering threshold before processing
# (This should be done before running video processing)
state_manager.set_item('face_clustering_threshold', 0.45)  # Try different values: 0.35, 0.4, 0.45, 0.5

# Example 2: After processing, inspect the database
print("\n=== Database Summary ===")
summary = get_database_summary()
if summary:
    print(f"Total clusters: {summary['total_clusters']}")
    print(f"Total faces: {summary['total_faces']}")
    print(f"Threshold used: {summary['clustering_threshold']}")
    print(f"Largest cluster: {summary['largest_cluster']} instances")
    print(f"Smallest cluster: {summary['smallest_cluster']} instances")
    print(f"Average cluster size: {summary['average_cluster_size']:.1f}")

# Example 3: Save the database for later inspection
save_database('face_database.pkl')

# Example 4: Load a previously saved database
# load_database('face_database.pkl')

# Example 5: Save face crops for all clusters (requires video path)
# video_path = '/path/to/your/video.mp4'
# save_cluster_face_crops(video_path, 'face_crops/')

# Example 6: Inspect specific cluster
clusters = get_all_clusters()
if clusters:
    cluster_0 = get_cluster(0)
    if cluster_0:
        print(f"\n=== Cluster 0 Details ===")
        print(f"Instances: {cluster_0['instance_count']}")
        print(f"Frame range: {cluster_0['frame_range']}")
        
        # Show spatial info from first instance
        if cluster_0['all_instances']:
            inst = cluster_0['all_instances'][0]
            print(f"Bounding box: {inst['bounding_box']}")
            print(f"Center: ({inst['center_x']:.1f}, {inst['center_y']:.1f})")
            print(f"Size: {inst['width']:.1f} x {inst['height']:.1f}")

print("\nDone!")

