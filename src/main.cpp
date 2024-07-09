#include <ros/ros.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Eigenvalues>
#include <string>
#include <glog/logging.h>
#include <chrono>
#include <ctime>
#include <sys/stat.h>

#include "/home/sgg-whu/TLS/EVPA/src/EVPA/include/tools.h"
#include "/home/sgg-whu/TLS/EVPA/src/EVPA/include/io_manager.h"
#include "EVPA.h"

using namespace std;

ros::Publisher pub_path,pub_initiate_map,pub_refined_map,pub_voxel,pub_plane,pub_notplane;
std::vector< std::shared_ptr <ros::Publisher> > pub_pointclouds_init;
std::vector< std::shared_ptr <ros::Publisher> > pub_pointclouds_refined;

enum PointCloudState {
    initial = 0,
    refined = 1
    };


// void ShowPointCloud(std::vector<PointCloudPtr> &pcd,std::vector<trajectory> &traj,PointCloudState &state,ros::Publisher&pub , ros::NodeHandle& n)
// {
    
//     PointCloudPtr pl_send(new PointCloud);
//     for(int i =0;i<pcd.size();i++)
//     {
//         PointCloudPtr temp(new PointCloud);
//         PointCloudPtr temp_t(new PointCloud);
//         sampleLeafsized(pcd[i],*temp,0.05);
//         tool::TransformPointCloud(temp,temp_t,traj[i]);
//         *pl_send+=*temp_t;
//         // if(pub[i]==nullptr)
//         // {
//         //     if(state == PointCloudState::initial)
//         //         pub[i] = std::make_shared<ros::Publisher>(n.advertise<sensor_msgs::PointCloud2>(std::string("/map_init").append(std::to_string(i)),100));
//         //     else if(state == PointCloudState::refined)
//         //         pub[i] = std::make_shared<ros::Publisher>(n.advertise<sensor_msgs::PointCloud2>(std::string("/map_refined").append(std::to_string(i)),100));
//         // }
//         // pub[i]->publish(out);

        
//     }
//     sensor_msgs::PointCloud2 out;
//     pcl::toROSMsg(*pl_send,out);
//     out.header.frame_id="init";
//     out.header.stamp = ros::Time::now();
//     pub.publish(out);
// }


int main(int argc,char**argv)
{
    ros::init(argc,argv,"EVPA");
    ros::NodeHandle n;
    pub_path = n.advertise<sensor_msgs::PointCloud2>("/map_path",100000);
    pub_initiate_map = n.advertise<sensor_msgs::PointCloud2>("/map_init",100000);
    pub_refined_map = n.advertise<sensor_msgs::PointCloud2>("/map_refined",100000);
    pub_voxel = n.advertise<visualization_msgs::Marker>("/voxel",100000);
    pub_plane = n.advertise<sensor_msgs::PointCloud2>("/plane",100000);
    pub_notplane = n.advertise<sensor_msgs::PointCloud2>("/not_plane",100000);
    std::string pcd_path,trajectory_path,save_traj_path;
    
    double voxel_size;
    double minmum_voxel_size =0.5;
    double voxel_decrease_step = 0.1;
    double down_sample_size;
    int reliable_region=3;
    int iter = 5;

    n.param<std::string>("pcd_path",pcd_path,"/home/sgg-whu/TLS/EVPA/src/data");
    n.param<std::string>("trajectory_path",trajectory_path,"/home/sgg-whu/TLS/EVPA/src/data/alidarPose.csv");
    n.param<std::string>("save_path",save_traj_path,"/home/sgg-whu/TLS/EVPA/src/HeritageBuilding");
    n.param<double>("voxel_size",voxel_size,1.0);
    n.param<double>("downsample_size",down_sample_size,0.05);
    n.param<double>("voxel_decrease_step",voxel_decrease_step,0.1);
    n.param<int>("iteration",iter,1);
    n.param<int>("reliable_region",reliable_region,3);
    n.param<double>("minmum_voxel_size",minmum_voxel_size,0.5);

    std::chrono::system_clock::time_point now =std::chrono::system_clock::now();
    std::time_t time_now = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_now = std::localtime(&time_now);
    std::string current_dir_name = std::to_string(tm_now->tm_year+1900)+"-"+std::to_string(tm_now->tm_mon)+"-"+std::to_string(tm_now->tm_mday)+"-"+std::to_string(tm_now->tm_hour)+":"+std::to_string(tm_now->tm_min)+":"+std::to_string(tm_now->tm_sec);

    std::string current_dir_name_full = save_traj_path+"/"+current_dir_name;

    std::vector<trajectory> traj;
    std::vector<PointCloudPtr> pcs;
    int pose_size = 0;
    int pcd_size = 0;

    //create file from current time

    mkdir(current_dir_name_full.c_str(),0777);

    //log initiate
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = current_dir_name_full;
    FLAGS_alsologtostderr=true;
    FLAGS_stderrthreshold = google::INFO;

    //data preprocess
    pose_size = io::ReadTrajectory(traj,trajectory_path);
    pcd_size = io::ReadPointCloud(pcs,pcd_path,pose_size,down_sample_size);
    LOG(INFO)<<"pose size: "<<pose_size<<"\n";
    //show
    if(pose_size!=pcd_size)
    {
        LOG(FATAL)<<"trajectories do not match point clouds: \n"<<pose_size<<"<->"<<pcd_size;
        return -1;
    }
    LOG(INFO)<<"voxel size :"<<voxel_size<<"\n";
    LOG(INFO)<<"downsample_size:"<<down_sample_size<<"\n";
    LOG(INFO)<<"reliable_region:"<<reliable_region<<"\n";
    //LOG(INFO)<<"iteration :"<<iter<<"\n";
    PointCloudState state = PointCloudState::initial;
    // ShowPointCloud(pcs,traj,state,pub_initiate_map,n);
    LOG(INFO)<<"point cloud loaded...\n";

    // EVPA process begin
    EVPA evpa;

    evpa.SetPoseSize(pose_size);
    evpa.SetReliableRegion(reliable_region);
    //evpa.VisualizePointCloud(pcs,traj);
    //evpa.SetNumThread(8);
    int i =0;
    while(evpa.solve_flag)
    {
        if((voxel_size-voxel_decrease_step)>minmum_voxel_size)
            voxel_size-=voxel_decrease_step;
        else voxel_size = minmum_voxel_size;
        LOG(INFO)<<"----------iter "<<i+1<<"----------\n";
        LOG(INFO)<<"voxel_size: "<<voxel_size<<"\n";
        evpa.init();
        evpa.SetVoxelSize(voxel_size);
        for (int i = 0; i < pose_size; i++)
        {
            evpa.CutVoxel(pcs[i], traj[i], i);
        }


        evpa.SolveWithFactorGraph(pub_voxel, pub_plane, pub_notplane, traj);

        evpa.PoseGraphOptimization(traj);

        //evpa.SolveWithLum(pcs,traj);
        // io::SaveTrajectoryKitti(traj,current_dir_name_full+"/refined_trajectory_kitti_iter"+std::to_string(i)+".txt");
        // io::SaveTrajectory(traj,current_dir_name_full+"/refined_trajectory_iter"+std::to_string(i)+".txt");
        //std::cout<<"--------------------------\n";
        io::SaveTrajectoryEVPA(traj,current_dir_name_full+"/refined_trajectory_evpa_iter"+std::to_string(i)+".txt");
        io::SaveTrajectoryKitti(traj,current_dir_name_full+"/refined_trajectory_kitti"+std::to_string(i)+".txt");

        i++;
    }

    // io::SaveTrajectoryKitti(traj,current_dir_name_full+"/refined_trajectory_kitti"+std::to_string(i)+".txt");
    // io::SaveTrajectory(traj,current_dir_name_full+"/refined_trajectory"+std::to_string(i)+".txt");
    // for(int i =0;i<iter;i++)
    // {
    //     LOG(INFO)<<"----------iter "<<i+1<<"----------\n";
    //     LOG(INFO)<<"voxel_size: "<<voxel_size<<"\n";
    //     evpa.init();
    //     evpa.SetVoxelSize(voxel_size);
    //     for (int i = 0; i < pose_size; i++)
    //     {
    //         evpa.CutVoxel(pcs[i], traj[i], i);
    //     }


    //     evpa.SolveWithICP(pcs,traj);
    //     //evpa.SolveWithICPNormal(pcs,traj);
    //     //evpa.SolveWithGICP(pcs,traj);
    //     //evpa.VisualizePlaneMatch(pcs,traj);

    //     evpa.PoseGraphOptimization(traj);

    //     //evpa.SolveWithLum(pcs,traj);
    //     io::SaveTrajectoryKitti(traj,current_dir_name_full+"/refined_trajectory_kitti_iter"+std::to_string(i)+".txt");
    //     // io::SaveTrajectory(traj,current_dir_name_full+"/refined_trajectory_iter"+std::to_string(i)+".txt");
    //     //std::cout<<"--------------------------\n";
    //     io::SaveTrajectoryEVPA(traj,current_dir_name_full+"/refined_trajectory_evpa_iter"+std::to_string(i)+".txt");
    // }
    io::SaveTrajectory(traj,current_dir_name_full+"/refined_trajectory"+std::to_string(i)+".txt");
    LOG(INFO)<<"total time cost:"<<evpa.cost_time;
    // EVPA process end
    //evpa.PrintRelativePoseGraph();
    //io::SaveTrajectory(traj,save_traj_path+"/refined_trajectory.txt");
    
    google::ShutdownGoogleLogging();
    
    

    return 0;
}