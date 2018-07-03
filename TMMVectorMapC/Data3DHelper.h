#pragma once

#ifndef DATA_3D_HELPER_H
#define DATA_3D_HELPER_H

#include "HeadFiles.h"
#include "HeadDetect.h"

extern COpenNI g_openNi;

class Data3DHelper
{
public:
	static int GetSizeInImageBySizeIn3D(const int iSizeIn3D, const int iDistance)
	{
		if (iDistance == 0 || iSizeIn3D == 0)
		{
			return 0;
		}

		static double dConstFactor = 0.0; // ��������ʾ�ռ��У����ȵľ������Ϊ1���׵�ֱ�߽�ͶӰ��һ�����ؿ�
		static bool bIsFactorComputed = false; // ��Ϊֻ���ڵ�һ�μ���һ�γ������Ժ��ټ��㣬�������������״̬

		if (!bIsFactorComputed)
		{
			// �����ͼ�ϼٶ�����
			//pPoint3D[2] = { { 0, 0, 1000 }, { 100, 0, 1000 } }; // {��, ��, ���}
			//float X1, X2, Y1, Y2, D1, D2;
			//float rX1, rX2, rY1, rY2, rZ1, rZ2;
			//X1 = 0; //1����
			//Y1 = 0; //1����
			//D1 = 1000;
			//X2 = 100; //2����
			//Y2 = 0; //2����
			//D2 = 1000;
			//g_openNi.m_DepthGenerator.ConvertProjectiveToRealWorld(2, pPoint3D, pPoint3D); // ���Եõ����ڿռ��е�ʵ������
			//CoordinateConverter::convertDepthToWorld(g_openNi.streamDepth, X1, Y1, D1, &rX1, &rY1, &rZ1);
			//CoordinateConverter::convertDepthToWorld(g_openNi.streamDepth, X2, Y2, D2, &rX2, &rY2, &rZ2);
			// �����ڿռ��е�ʵ�ʾ���
			double d3DDistance = 173.66668978426750;//std::sqrt((long double)((rX1 - rX2) * (rX1 - rX2) + (rY1 - rY2) * (rY1 - rY2) + (rZ1 - rZ2) * (rZ1 - rZ2)));
			dConstFactor = d3DDistance / (1000 * 100);
			bIsFactorComputed = true;
		}

		return (int)(((double)iSizeIn3D / (double)iDistance) / dConstFactor);

	}

};

#endif