#include "io_manager.h"

int io::ReadTrajectory(std::vector<trajectory> &traj, std::string path)
{
    std::ifstream ifs(path);
    if(!ifs.is_open())
    {
        std::cout<<"can not open file!\n";
        return 0;
    }

    int pose_size = 0;
    std::string line_string,temp_string;
    int lines = 0;
    Eigen::Matrix4d trans;
    std::vector<double> nums;

    while(getline(ifs,line_string))
    {
        lines++;
        std::stringstream ss(line_string);
        while(getline(ss,temp_string,','))
            nums.push_back(stod(temp_string));
        
        if(lines ==4)
        {
            for(int j=0;j<16;j++)
                trans(j) = nums[j];
            
            Eigen::Matrix4d trans_T = trans.transpose();
            //std::cout<<"rotation: "<<trans_T<<"\n";
            Eigen::Quaternion<double> trans_q(trans_T.block<3,3>(0,0));
            Eigen::Vector3d translation(trans_T.block<3,1>(0,3));                              
            trajectory curr;
            curr.para_q[0] =trans_q.x();
            curr.para_q[1] =trans_q.y();
            curr.para_q[2] =trans_q.z();
            curr.para_q[3] =trans_q.w();
            //std::cout<<"quaternion: "<<trans_q.toRotationMatrix()<<"\n";
            curr.para_t[0] = translation(0);
            curr.para_t[1] = translation(1);
            curr.para_t[2] = translation(2);
            //std::cout<<"translation:" <<curr.para_t[0]<<","<<curr.para_t[1]<<","<<curr.para_t[2]<<"\n";

            traj.push_back(curr);
            nums.clear();
            lines = 0;
            pose_size++;
        }

    }
    return pose_size;

}

int io::ReadPointCloud(std::vector<PointCloudPtr> &pcs, std::string path,int pose_size,float down_sample_size)
{
    struct dirent *ptr;
    DIR *dir;

    dir = opendir(path.c_str());
    std::vector<std::string> files;
    int file_size= 0;

    while((ptr=readdir(dir))!=0)
    {
        if(std::strcmp(ptr->d_name,".")==0 || std::strcmp(ptr->d_name,"..")==0 )
            continue;
        
        files.push_back(ptr->d_name);

    }
    std::sort(files.begin(),files.end());

    for(int i=0;i<files.size();i++)
    {
        std::cout<<files[i]<<"\n";
    }

    for(int i=0;i<files.size();i++)
    {
        PointCloudPtr temp_pc(new PointCloud());
        PointCloudPtr temp_pc_d(new PointCloud());
        if(files[i].rfind(".pcd")!=std::string::npos)
        {
            pcl::io::loadPCDFile(path+"/"+files[i],*temp_pc);
            if (down_sample_size > 0)
            {
                tool::sampleLeafsized(temp_pc, *temp_pc_d, down_sample_size);
                pcs.push_back(temp_pc_d);
            }
            else{
                pcs.push_back(temp_pc);
            }
            file_size++;
        }
        else if(files[i].rfind(".ply")!=std::string::npos)
        {
            pcl::io::loadPLYFile(path+"/"+files[i],*temp_pc);
            if (down_sample_size > 0)
            {
                tool::sampleLeafsized(temp_pc, *temp_pc_d, down_sample_size);
                pcs.push_back(temp_pc_d);
            }
            else{
                pcs.push_back(temp_pc);
            }
            //pcs.push_back(temp_pc_d);
            file_size++;
        }
        //std::cout<<temp_pc->points.size()<<"\n";
    }
    closedir(dir);

    return file_size;
}

bool io::SaveTrajectoryKitti(std::vector<trajectory> &traj, std::string path)
{
    std::ofstream ofs(path);

    if(!ofs.is_open())
    {
        LOG(ERROR)<<"can not open file!\n";
        return false;
    }

    for(int i =0;i<traj.size();i++)
    {
        Eigen::Quaterniond q_i(traj[i].para_q[3],traj[i].para_q[0],traj[i].para_q[1],traj[i].para_q[2]);
        Eigen::Vector3d t_i(traj[i].para_t[0],traj[i].para_t[1],traj[i].para_t[2]);

        Eigen::Affine3d se3;
        se3.linear() = q_i.toRotationMatrix();
        se3.translation() = t_i;
        Eigen::IOFormat evpa_format(Eigen::StreamPrecision,Eigen::DontAlignCols," "," ","","");
        ofs << se3.matrix().block<3,4>(0,0).format(evpa_format);
        ofs <<"\n";
    }
    return true;
}
bool io::SaveTrajectory(std::vector<trajectory> &traj,std::string path)
{
    std::ofstream ofs(path);

    if(!ofs.is_open())
    {
        LOG(ERROR)<<"can not open file!\n";
        return false;
    }

    for(int i =0;i<traj.size();i++)
    {
        Eigen::Quaterniond q_i(traj[i].para_q[3],traj[i].para_q[0],traj[i].para_q[1],traj[i].para_q[2]);
        Eigen::Vector3d t_i(traj[i].para_t[0],traj[i].para_t[1],traj[i].para_t[2]);

        Eigen::Affine3d se3;
        se3.linear() = q_i.toRotationMatrix();
        se3.translation() = t_i;
        //Eigen::IOFormat evpa_format(Eigen::StreamPrecision,Eigen::DontAlignCols," "," ","","");
        ofs << se3.matrix();
        ofs <<"\n";
    }
    return true;
}


bool io::SavePointcloud(std::unordered_map<VoxelLocation,Voxel*> feature_map, std::string path)
{
    PointCloudPtr cloud(new PointCloud);
    for(auto iter =feature_map.begin();iter!=feature_map.end();iter++)
    {
        for(int i=0;i<iter->second->pts_trans.size();i++)
        {
            if(iter->second->pts_trans[i].size()==0)
                continue;

            for(int  j=0;j<iter->second->pts_trans[i].size();j++)
            {
                PointType pts(iter->second->pts_trans[i][j].x(),iter->second->pts_trans[i][j].y(),iter->second->pts_trans[i][j].z());

                cloud->points.push_back(pts);
            }
        }
    }

    
}

bool io::SaveTrajectoryEVPA(std::vector<trajectory> &traj,std::string path)
{
    std::ofstream ofs(path);

    if(!ofs.is_open())
    {
        LOG(ERROR)<<"can not open file!\n";
        return false;
    }

    for(int i =0;i<traj.size();i++)
    {
        Eigen::Quaterniond q_i(traj[i].para_q[3],traj[i].para_q[0],traj[i].para_q[1],traj[i].para_q[2]);
        Eigen::Vector3d t_i(traj[i].para_t[0],traj[i].para_t[1],traj[i].para_t[2]);

        Eigen::Affine3d se3;
        se3.linear() = q_i.toRotationMatrix();
        se3.translation() = t_i;
        Eigen::IOFormat evpa_format(Eigen::FullPrecision,0,",","\n","",",");
        ofs << se3.matrix().format(evpa_format);
        ofs <<"\n";
    }
}
