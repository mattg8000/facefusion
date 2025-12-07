#!/usr/bin/env python3
"""
Test script to inspect the video face database.
Run this in a Python shell after processing a video to query the database.

Usage:
    # After processing a video, in Python shell:
    exec(open('test_database.py').read())
    
    # Or import and use:
    from test_database import print_database_info, query_cluster
    print_database_info()
"""

from facefusion.video_face_database import (
    get_video_face_database,
    get_database_summary,
    get_all_clusters,
    get_cluster,
    get_cluster_for_face,
    save_database,
    load_database,
    get_cluster_face_crop,
    save_cluster_face_crops
)

def print_database_info():
    """Print database information"""
    database = get_video_face_database()
    
    if database is None:
        print("❌ No face database found.")
        print("   Run video processing first to create the database.")
        return
    
    summary = get_database_summary()
    clusters = get_all_clusters()
    
    print("\n" + "=" * 70)
    print("VIDEO FACE DATABASE")
    print("=" * 70)
    print(f"Total Clusters: {summary['total_clusters']}")
    print(f"Total Face Instances: {summary['total_faces']}")
    print(f"Clustering Threshold: {summary['clustering_threshold']}")
    print(f"Largest Cluster: {summary['largest_cluster']} instances")
    print(f"Smallest Cluster: {summary['smallest_cluster']} instances")
    print(f"Average Cluster Size: {summary['average_cluster_size']:.1f} instances")
    print()
    
    # Sort by size
    sorted_clusters = sorted(clusters, key=lambda c: c['instance_count'], reverse=True)
    
    print("Top 10 Largest Clusters:")
    print("-" * 70)
    for i, cluster in enumerate(sorted_clusters[:10], 1):
        frame_span = cluster['frame_range'][1] - cluster['frame_range'][0] + 1
        unique_frames = len(set([inst['frame_number'] for inst in cluster['all_instances']]))
        print(f"{i:2}. Cluster {cluster['cluster_id']:2}: {cluster['instance_count']:4} instances, "
              f"frames {cluster['frame_range'][0]}-{cluster['frame_range'][1]} "
              f"({unique_frames} unique frames)")
    
    print()
    print("=" * 70)
    print("QUERY FUNCTIONS")
    print("=" * 70)
    print("Available functions:")
    print("  get_database_summary() - Get summary statistics")
    print("  get_all_clusters() - Get all clusters")
    print("  get_cluster(cluster_id) - Get specific cluster")
    print("  get_cluster_for_face(frame_number, face_index) - Find cluster for a face")
    print("  save_database(file_path) - Save database to file")
    print("  load_database(file_path) - Load database from file")
    print("  get_cluster_face_crop(video_path, cluster_id) - Get face crop for cluster")
    print("  save_cluster_face_crops(video_path, output_dir) - Save all cluster face crops")
    print()
    print("Example:")
    print("  cluster = get_cluster(0)  # Get first cluster")
    print("  cluster_id = get_cluster_for_face(100, 0)  # Get cluster for frame 100, face 0")
    print("  save_database('face_db.pkl')  # Save database")
    print("  save_cluster_face_crops('video.mp4', 'face_crops/')  # Save face images")
    print()
    
    # Show cluster size distribution
    size_ranges = {
        '< 10': 0,
        '10-49': 0,
        '50-99': 0,
        '100+': 0
    }
    for cluster in clusters:
        size = cluster['instance_count']
        if size < 10:
            size_ranges['< 10'] += 1
        elif size < 50:
            size_ranges['10-49'] += 1
        elif size < 100:
            size_ranges['50-99'] += 1
        else:
            size_ranges['100+'] += 1
    
    print("Cluster Size Distribution:")
    for size_range, count in size_ranges.items():
        if count > 0:
            print(f"  {size_range}: {count} clusters")


def query_cluster(cluster_id: int):
    """Query a specific cluster and show details"""
    cluster = get_cluster(cluster_id)
    if cluster is None:
        print(f"Cluster {cluster_id} not found")
        return
    
    print(f"\nCluster {cluster_id} Details:")
    print(f"  Instances: {cluster['instance_count']}")
    print(f"  Frame Range: {cluster['frame_range'][0]} - {cluster['frame_range'][1]}")
    
    # Show frame distribution
    frame_numbers = sorted(set([inst['frame_number'] for inst in cluster['all_instances']]))
    print(f"  Appears in {len(frame_numbers)} unique frames")
    print(f"  Sample frames: {frame_numbers[:10]}{'...' if len(frame_numbers) > 10 else ''}")
    
    # Show face properties from representative
    rep_face = cluster['representative_face']
    print(f"  Representative face:")
    print(f"    Gender: {rep_face.gender}")
    print(f"    Age: {rep_face.age}")
    print(f"    Race: {rep_face.race}")
    print(f"    Detector score: {rep_face.score_set.get('detector', 0):.3f}")


if __name__ == '__main__':
    print_database_info()

if __name__ == '__main__':
    print_database_info()

