#ifndef IO_MANAGER_H
#define IO_MANAGER_H
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <dirent.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include "tools.h"
#include "EVPA.h"


class io
{
public:
    static int ReadTrajectory(std::vector<trajectory> &traj,std::string path);
    static int ReadPointCloud(std::vector<PointCloudPtr> &pcs,std::string path,int pose_size,float down_sample_size);
    static bool SaveTrajectoryKitti(std::vector<trajectory> &traj,std::string path);
    static bool SaveTrajectory(std::vector<trajectory> &traj,std::string path);
    static bool SaveTrajectoryEVPA(std::vector<trajectory> &traj,std::string path);
    static bool SavePointcloud(std::unordered_map<VoxelLocation,Voxel*> feature_map,std::string path);
};


#endif