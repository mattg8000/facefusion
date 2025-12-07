#!/usr/bin/env python3
"""
Simple script to inspect the video face database after preprocessing.
Run this after processing a video to see cluster information.
"""

import sys
from facefusion.video_face_database import get_video_face_database, get_database_summary, get_all_clusters, get_cluster

def main():
    database = get_video_face_database()
    
    if database is None:
        print("No face database found. Run video processing first to create the database.")
        return
    
    summary = get_database_summary()
    clusters = get_all_clusters()
    
    print("=" * 60)
    print("VIDEO FACE DATABASE SUMMARY")
    print("=" * 60)
    print(f"Total Clusters: {summary['total_clusters']}")
    print(f"Total Face Instances: {summary['total_faces']}")
    print(f"Clustering Threshold: {summary['clustering_threshold']}")
    print(f"Largest Cluster: {summary['largest_cluster']} instances")
    print(f"Smallest Cluster: {summary['smallest_cluster']} instances")
    print(f"Average Cluster Size: {summary['average_cluster_size']:.1f} instances")
    print()
    
    print("=" * 60)
    print("CLUSTER DETAILS")
    print("=" * 60)
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(clusters, key=lambda c: c['instance_count'], reverse=True)
    
    for cluster in sorted_clusters:
        print(f"\nCluster {cluster['cluster_id']}:")
        print(f"  Instances: {cluster['instance_count']}")
        print(f"  Frame Range: {cluster['frame_range'][0]} - {cluster['frame_range'][1]}")
        print(f"  Frames Span: {cluster['frame_range'][1] - cluster['frame_range'][0] + 1} frames")
        
        # Show sample frame numbers
        sample_frames = sorted(set([inst['frame_number'] for inst in cluster['all_instances'][:10]]))
        if len(sample_frames) > 5:
            print(f"  Sample Frames: {sample_frames[:5]} ... (showing first 5 of {len(set([inst['frame_number'] for inst in cluster['all_instances']]))} unique frames)")
        else:
            print(f"  Sample Frames: {sample_frames}")
    
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    # Check for potential issues
    if summary['total_clusters'] > summary['total_faces'] * 0.1:
        print(f"⚠️  WARNING: Many small clusters detected ({summary['total_clusters']} clusters)")
        print(f"   This suggests clustering threshold ({summary['clustering_threshold']}) may be too strict.")
        print(f"   Consider increasing threshold to 0.4-0.5 to merge similar faces.")
    
    # Check for very small clusters (likely noise or false detections)
    small_clusters = [c for c in clusters if c['instance_count'] < 5]
    if small_clusters:
        print(f"\n⚠️  Found {len(small_clusters)} very small clusters (< 5 instances)")
        print(f"   These might be false detections or edge cases.")
        print(f"   Small clusters: {[c['cluster_id'] for c in small_clusters]}")
    
    # Check cluster size distribution
    size_distribution = {}
    for cluster in clusters:
        size = cluster['instance_count']
        if size < 10:
            size_range = "< 10"
        elif size < 50:
            size_range = "10-49"
        elif size < 100:
            size_range = "50-99"
        else:
            size_range = "100+"
        size_distribution[size_range] = size_distribution.get(size_range, 0) + 1
    
    print(f"\nCluster Size Distribution:")
    for size_range, count in sorted(size_distribution.items()):
        print(f"  {size_range}: {count} clusters")

if __name__ == '__main__':
    main()

