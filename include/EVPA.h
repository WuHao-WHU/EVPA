#ifndef EVPA_H
#define EVPA_H
#include <unordered_map>
#include <Eigen/Dense>
#include "tools.h"
#include <ceres/ceres.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/impl/filter.hpp>
#include <pcl/registration/icp.h>
#include <pcl/registration/lum.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <omp.h>
#include <memory>


class VoxelLocation
{
public:
    int64_t x,y,z;

    VoxelLocation(int64_t vx=0, int64_t vy=0,int64_t vz=0):x(vx),y(vy),z(vz){}

    bool operator == (const VoxelLocation &other ) const
    {
        return (x==other.x &&y==other.y && z== other.z);
    }
};

namespace std
{
    template<>
    struct hash<VoxelLocation>
    {
        size_t operator() (const VoxelLocation &s) const
        {
            using std::size_t;using std::hash;
            return (((std::hash<int64_t>()(s.z)*HASH_P)%MAX_N +std::hash<int64_t>()(s.y))*HASH_P)%MAX_N +std::hash<int64_t>()(s.x);   
        }
    };
}

struct feature
{
    int N;
    Eigen::Matrix3d P;
    Eigen::Vector3d v;

    feature():N(0){
        P.setZero();
        v.setZero();
    }

    void push(const Eigen::Vector3d &vec)
    {
        N++;
        P +=vec*vec.transpose();
        v += vec;
    }

    Eigen::Matrix3d cov()
    {
        Eigen::Vector3d center = v/N;
        return P/N - center*center.transpose();
    }

    feature & operator+=(const feature & feat)
    {
        this->P+=feat.P;
        this->v+=feat.v;
        this->N+=feat.N;

        return *this;
    }

};

class Voxel
{
public:
    //0-point to point factor
    //1-point to plane factor
    int push_state;
    //point cloud per scan
    //feature per scan
    std::vector<std::vector<Eigen::Vector3d>> pts_orig,pts_trans;
    std::vector<feature> feat_orig,feat_trans;
    float voxel_center[3];
    double decision;
    Eigen::Vector3d value_vector;

public:    
    Voxel(int pose_size)
    {
        push_state = 0;
        pts_orig.resize(pose_size);
        pts_trans.resize(pose_size);
        feat_orig.resize(pose_size);
        feat_trans.resize(pose_size);
    }

    void PushBack(Eigen::Vector3d &p_orig,Eigen::Vector3d &p_trans,int n_scan)
    {
        pts_orig[n_scan].push_back(p_orig);
        pts_trans[n_scan].push_back(p_trans);
        feat_orig[n_scan].push(p_orig);
        feat_trans[n_scan].push(p_trans);
    }

    bool JudgePlane(int n_scan)
    {
        feature cov_mat;
        for(int i =0;i<n_scan;i++)
            cov_mat+=feat_trans[i];

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_mat.cov());

        value_vector = saes.eigenvalues();

        decision = saes.eigenvalues()[0]/saes.eigenvalues()[1];

        return decision<(1.0/36);
    }
};



class EVPA
{
public:

    EVPA(){
        voxel_size = 0.5;
        num_threads = 1;
        //use_point2point= true;
    }

    void SetVoxelSize(double voxel_size){this->voxel_size = voxel_size;};
    void SetPoseSize(int pose_size){this->pose_size=pose_size;};
    void SetNumThread(int num_threads){this->num_threads = num_threads;};
    void SetReliableRegion(int reliable_region){this->reliable_region = reliable_region;};
    void DisplayVoxel(ros::Publisher &pub_voxel,ros::Publisher &pub_plane,ros::Publisher &pub_notplane);

    void CutVoxel(PointCloudPtr &pc,trajectory &traj,int n_scan);

    void SolveWithFactorGraph(ros::Publisher &pub_voxel,ros::Publisher &pub_plane,ros::Publisher &pub_notplane,std::vector<trajectory> &traj_s);


    void SolveWithICP(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s);
    void SolveWithICPNormal(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s);
    void SolveWithLum(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s);
    void SolveWithGICP(std::vector<PointCloudPtr> &pcs,std::vector<trajectory> &traj_s);

    void PrintRelativePoseGraph();

    void PoseGraphOptimization(std::vector<trajectory> &traj_s);

    void init();

    void VisualizePlaneMatch(std::vector<PointCloudPtr> &pc,std::vector<trajectory> &traj_s);

    void VisualizePointCloud(std::vector<PointCloudPtr> &pc,std::vector<trajectory> &traj_s);
    //void SlideWindow();
    


public:

    //<pose index, corresponding pose>
    std::unordered_map<int,trajectory> relative_pose;
    std::unordered_map<int,trajectory> relative_pose_point;
    std::unordered_map<int,trajectory> relative_pose_point2point;
    //bool use_point2point;
    //<pose index, corresponding col,raw
    // register col -> raw
    std::unordered_map<int,std::pair<int,int>> relative_pose_graph;
    std::unordered_map<VoxelLocation,Voxel*> Voxelized_map;
    std::vector<Voxel*> Voxelized_map_plane_visualize;
    std::vector<std::vector<int>> pose_index_each_voxel;
    double voxel_size;
    int pose_size;
    int num_threads;
    int overlap_count = 0;
    int plane_count = 0;
    int notplane_count = 0;
    int last_plane_count =0;

    bool solve_flag=true;
    int reliable_region=3;
    double cost_time = 0;
    // std::vector<std::pair<const double*,const double*>> covariance_blocks;
    // int 
    // int window_size = 3;
};


#endif