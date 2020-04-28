//
//  main.c
//  Player06
//
//  Created by 侯森魁 on 2020/4/25.
//  Copyright © 2020 侯森魁. All rights reserved.
//

#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "SDL.h"

#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"

// compatibility with newer API
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif

#define SDL_AUDIO_BUFFER_SIZE 1024
#define MAX_AUDIO_FRAME_SIZE 192000 //channels(2) * data_size(2) * sample_rate(48000)

#define MAX_AUDIOQ_SIZE (5 * 16 * 1024)
#define MAX_VIDEOQ_SIZE (5 * 256 * 1024)

#define AV_SYNC_THRESHOLD 0.01
#define AV_NOSYNC_THRESHOLD 10.0

#define SAMPLE_CORRECTION_PERCENT_MAX 10
#define AUDIO_DIFF_AVG_NB 20

#define FF_REFRESH_EVENT (SDL_USEREVENT)
#define FF_QUIT_EVENT (SDL_USEREVENT + 1)

#define VIDEO_PICTURE_QUEUE_SIZE 1

#define DEFAULT_AV_SYNC_TYPE AV_SYNC_AUDIO_MASTER //默认的视音频同步方式，采用视频时钟同步到音频时钟的方式

#define PLAYER_WIDTH 1080 //播放器宽度

#define PLAYER_HEIGHT 608 //播放器高度

#define  DEBUG1 0

typedef struct PacketQueue {
    AVPacketList *first_pkt, *last_pkt; //队列中第一个pkt和最后一盒pkt
    int nb_packets;//队列中有多少个packet
    int size;  //总字节大小
    SDL_mutex *mutex; //互斥
    SDL_cond *cond; //查看源码、有示例
} PacketQueue;


typedef struct VideoPicture {
    AVPicture *bmp;
    int width, height; /* source height & width */
    int allocated;
    double pts;
} VideoPicture;

typedef struct VideoState {
    
    //multi-media file
    char            filename[1024];
    AVFormatContext *pFormatCtx;
    int             videoStream, audioStream;
    
    //sync
    int             av_sync_type;
    
    double          audio_diff_cum; /* used for AV difference average computation */
    double          audio_diff_avg_coef; //音频差异平均系数
    double          audio_diff_threshold;//音频差异阈值
    int             audio_diff_avg_count;//音频差异平均计数
    
    
    double          audio_clock;//音频正在播放的时间
    double          frame_timer; //下一次timer回调的时间
    double          frame_last_pts;//上一次视频帧pts的时间
    double          frame_last_delay;//上一次视频帧增加的delay时间
    
    double          video_clock; ///<pts of last decoded frame / predicted pts of next decoded frame 下一帧视频将要播放的时间
    double          video_current_pts; ///<current displayed pts (different from video_clock if frame fifos are used)
    int64_t         video_current_pts_time;  ///<time (av_gettime) at which we updated video_current_pts - used to have running video pts
    
    //audio
    AVStream        *audio_st; //音频流
    AVCodecContext  *audio_ctx;//音频解码的上下文
    PacketQueue     audioq;//音频队列
    uint8_t         audio_buf[(MAX_AUDIO_FRAME_SIZE * 3) / 2];//解码后的音频缓冲区
    unsigned int    audio_buf_size;//缓冲区的大小
    unsigned int    audio_buf_index;//现在已经使用了多少字节
    AVFrame         audio_frame;//解码后的音频帧
    AVPacket        audio_pkt;//解码之前的音频包
    uint8_t         *audio_pkt_data; //解码之前音频包的具体数据的指针
    int             audio_pkt_size;//解码之前音频包的具体数据的包的大小
    int             audio_hw_buf_size; // SDL音频缓冲区大小(单位字节)
    struct  SwrContext *audio_swr_ctx; //音频重采样上下文
    //video
    AVStream        *video_st; //视频的流
    AVCodecContext  *video_ctx;//视频的上下文
    PacketQueue     videoq;//视频流队列
    struct SwsContext *video_sws_ctx; //视频图像裁剪、缩放上下文

    
    VideoPicture    pictq[VIDEO_PICTURE_QUEUE_SIZE]; //解码后的视频帧队列
    int             pictq_size, pictq_rindex, pictq_windex;//解码后的视频流队列的大小、获取视频帧的位置、存放视频帧的位置
    
    SDL_mutex       *pictq_mutex; //解码后的视频帧队列有一把锁
    SDL_cond        *pictq_cond; //解码后的视频帧队列的信号量
    
    SDL_Thread      *parse_tid;//解复用线程
    SDL_Thread      *video_tid;//解码线程
    
    int             quit; //结束SDL窗口标记
} VideoState;

SDL_mutex    *text_mutex;
SDL_Window   *win = NULL;
SDL_Renderer *renderer;
SDL_Texture  *texture;

enum {
    AV_SYNC_AUDIO_MASTER,
    AV_SYNC_VIDEO_MASTER,
    AV_SYNC_EXTERNAL_MASTER,
};

static int screen_left = SDL_WINDOWPOS_CENTERED;
static int screen_top = SDL_WINDOWPOS_CENTERED;
static int screen_width = 0;
static int screen_height = 0;
static int resize = 1;

FILE *audiofd = NULL;

/* Since we only have one decoding thread, the Big Struct
 can be global in case we need it. */
VideoState *global_video_state;

void packet_queue_init(PacketQueue *q) {
    memset(q, 0, sizeof(PacketQueue));
    q->mutex = SDL_CreateMutex();
    q->cond = SDL_CreateCond();
}
//向链表队列尾部插入节点
int packet_queue_put(PacketQueue *q, AVPacket *pkt) {
    
    AVPacketList *pkt1;
    if(av_dup_packet(pkt) < 0) {
        return -1;
    }
    pkt1 = av_malloc(sizeof(AVPacketList));
    if (!pkt1)
        return -1;
    pkt1->pkt = *pkt;
    pkt1->next = NULL;
    
    SDL_LockMutex(q->mutex);
    
    if (!q->last_pkt)
        q->first_pkt = pkt1;
    else
        q->last_pkt->next = pkt1; //将当前队列的最后的一个节点的next指针指向pkt1,
    q->last_pkt = pkt1; //将当前队列的最后一个节点设置为pkt1
    q->nb_packets++; //队列的packet数量加1
    q->size += pkt1->pkt.size; //队列的总字节大小 需要加上当前pkt1节点的大小
    SDL_CondSignal(q->cond); //使用锁每次只能一个线程进来
    
    SDL_UnlockMutex(q->mutex);
    return 0;
}

//取出队列中的首节点
int packet_queue_get(PacketQueue *q, AVPacket *pkt, int block)
{
    AVPacketList *pkt1;
    int ret;
    
    SDL_LockMutex(q->mutex);
    
    for(;;) {
        
        if(global_video_state->quit) {
            ret = -1;
            break;
        }
        
        pkt1 = q->first_pkt;
        if (pkt1) {
            q->first_pkt = pkt1->next;
            if (!q->first_pkt)
                q->last_pkt = NULL;
            q->nb_packets--;
            q->size -= pkt1->pkt.size;
            *pkt = pkt1->pkt;
            av_free(pkt1);
            ret = 1;
            break;
        } else if (!block) {
            ret = 0;
            break;
        } else {
            SDL_CondWait(q->cond, q->mutex);//等待
        }
    }
    SDL_UnlockMutex(q->mutex);
    return ret;
}

double get_audio_clock(VideoState *is) {
    double pts;
    int hw_buf_size, bytes_per_sec, n;
    
    pts = is->audio_clock; /* maintained in the audio thread */ /*上一步 获取的PTS*/
    hw_buf_size = is->audio_buf_size - is->audio_buf_index; //音频缓冲区中还没有播放的数据(单位字节)
    bytes_per_sec = 0;
    n = is->audio_ctx->channels * 2;
    if(is->audio_st) {
        bytes_per_sec = is->audio_ctx->sample_rate * n; //每秒钟播放的字节数
    }
    if(bytes_per_sec) {
        
        pts -= (double)hw_buf_size / bytes_per_sec; //PTS - (缓冲区里还要消耗的时间)
    }
#if DEBUG1
    printf("pts = %lf ----- \n",pts);
#endif
    return pts;
}


int audio_decode_frame(VideoState *is, uint8_t *audio_buf, int buf_size, double *pts_ptr) {
    
    int len1, data_size = 0;
    AVPacket *pkt = &is->audio_pkt;
    double pts;
    int n;
    
    for(;;) {
        while(is->audio_pkt_size > 0) {
            int got_frame = 0;
            //从解码之前音频包中取出packet进行解码
            len1 = avcodec_decode_audio4(is->audio_ctx, &is->audio_frame, &got_frame, pkt);
            if(len1 < 0) {
                /* if error, skip frame */
                is->audio_pkt_size = 0;
                break;
            }
            data_size = 0;
            if(got_frame) { //音频解码成功 需要进行重采样
                /*
                 data_size = av_samples_get_buffer_size(NULL,
                 is->audio_ctx->channels,
                 is->audio_frame.nb_samples,
                 is->audio_ctx->sample_fmt,
                 1);
                 */
                data_size = 2 * is->audio_frame.nb_samples * 2;
                assert(data_size <= buf_size);
                
                swr_convert(is->audio_swr_ctx,
                            &audio_buf,
                            MAX_AUDIO_FRAME_SIZE*3/2,
                            (const uint8_t **)is->audio_frame.data,
                            is->audio_frame.nb_samples);
                
                fwrite(audio_buf, 1, data_size, audiofd);
                //memcpy(audio_buf, is->audio_frame.data[0], data_size);
            }
            is->audio_pkt_data += len1;
            is->audio_pkt_size -= len1; //减去 解码成功的那部分数据
            if(data_size <= 0) {
                /* No data yet, get more frames */
                continue;
            }
            pts = is->audio_clock;
            *pts_ptr = pts;
            n = 2 * is->audio_ctx->channels;
            is->audio_clock += (double)data_size /
            (double)(n * is->audio_ctx->sample_rate);
//            printf("is->audio_clock = %lf ---- \n",is->audio_clock);
            /* We have data, return it and come back for more later */
            return data_size;
        }
        if(pkt->data)
            av_free_packet(pkt);
        
        if(is->quit) { //取下一个包之前要进行判断 是否要退出
            return -1;
        }
        /* next packet */
        if(packet_queue_get(&is->audioq, pkt, 1) < 0) { //从音频队列中取包
            return -1;
        }
        is->audio_pkt_data = pkt->data;
        is->audio_pkt_size = pkt->size;
        /* if update, update the audio clock w/pts */
        
        if(pkt->pts != AV_NOPTS_VALUE) {
            is->audio_clock = av_q2d(is->audio_st->time_base)*pkt->pts;
        }
#if DEBUG1
        printf("pkt->pts = %lld---\n",pkt->pts);
        printf("is->audio_st->time_base.num  = %d --- \n",is->audio_st->time_base.num);
        printf("is->audio_st->time_base.den  = %d ---- \n",is->audio_st->time_base.den);
        printf("av_q2d(is->audio_st->time_base) = %lf --- \n",av_q2d(is->audio_st->time_base));
        printf("is->audio_clock = %lf ---- \n",is->audio_clock);
#endif
    }
}
//回调函数，向SDL缓冲区填充数据
void audio_callback(void *userdata, Uint8 *stream, int len) {
    
    VideoState *is = (VideoState *)userdata;
    int len1, audio_size;
    double pts;
    
    SDL_memset(stream, 0, len);
    /*   len是由SDL传入的SDL缓冲区的大小，如果这个缓冲未满，我们就一直往里填充数据 */
     /*  audio_buf_index 和 audio_buf_size 标示我们自己用来放置解码出来的数据的缓冲区，*/
    /*   这些数据待copy到SDL缓冲区， 当audio_buf_index >= audio_buf_size的时候意味着我*/
    /*   们的缓冲为空，没有数据可供copy，这时候需要调用audio_decode_frame来解码出更多的桢数据 */
    while(len > 0) {
        if(is->audio_buf_index >= is->audio_buf_size) {
            /* We have already sent all our data; get more */
            audio_size = audio_decode_frame(is, is->audio_buf, sizeof(is->audio_buf), &pts);
            if(audio_size < 0) { /* audio_data_size < 0 标示没能解码出数据，我们默认播放静音 */
                /* If error, output silence */
                is->audio_buf_size = 1024 * 2 * 2;
                memset(is->audio_buf, 0, is->audio_buf_size);  /* 清零，静音 */
            } else {
                //注释
               /* audio_size = synchronize_audio(is, (int16_t *)is->audio_buf,
                                               audio_size, pts);*/
                is->audio_buf_size = audio_size;
            }
            is->audio_buf_index = 0;
        }
        /*  查看stream可用空间，决定一次copy多少数据，剩下的下次继续copy */
        len1 = is->audio_buf_size - is->audio_buf_index;
        if(len1 > len)
            len1 = len;
        SDL_MixAudio(stream,(uint8_t *)is->audio_buf + is->audio_buf_index, len1, SDL_MIX_MAXVOLUME);
        //memcpy(stream, (uint8_t *)is->audio_buf + is->audio_buf_index, len1);
        len -= len1;
        stream += len1;
        is->audio_buf_index += len1;
    }
}

static Uint32 sdl_refresh_timer_cb(Uint32 interval, void *opaque) {
    SDL_Event event;
    event.type = FF_REFRESH_EVENT;
    event.user.data1 = opaque;
    SDL_PushEvent(&event);
    return 0; /* 0 means stop timer */
}

/* schedule a video refresh in 'delay' ms */ /*每40毫秒触发主线程  安排视频刷新*/
static void schedule_refresh(VideoState *is, int delay) {
    SDL_AddTimer(delay, sdl_refresh_timer_cb, is);
}
//视频交给SDL库进行渲染
void video_display(VideoState *is) {
    
    SDL_Rect rect;
    VideoPicture *vp;
    float aspect_ratio;
    int w, h, x, y;
    int i;
    
    if(screen_width && resize){
//        SDL_SetWindowSize(win, screen_width, screen_height);
        SDL_SetWindowSize(win, PLAYER_WIDTH, PLAYER_HEIGHT); //修改的windows的尺寸，因为原始尺寸是视频帧中的大小，可能很大
        
        SDL_SetWindowPosition(win, screen_left, screen_top);
        SDL_ShowWindow(win);
        
        //IYUV: Y + U + V  (3 planes)
        //YV12: Y + V + U  (3 planes)
        Uint32 pixformat= SDL_PIXELFORMAT_IYUV;
        
        //create texture for render
        texture = SDL_CreateTexture(renderer,
                                    pixformat,
                                    SDL_TEXTUREACCESS_STREAMING,
                                    screen_width ,
                                    screen_height);
        resize = 0;
    }
    
    vp = &is->pictq[is->pictq_rindex];
    if(vp->bmp) {
        //将解码后的视频帧存放到纹理中去
        SDL_UpdateYUVTexture( texture, NULL,
                             vp->bmp->data[0], vp->bmp->linesize[0],
                             vp->bmp->data[1], vp->bmp->linesize[1],
                             vp->bmp->data[2], vp->bmp->linesize[2]);
        
        rect.x = 0;
        rect.y = 0;
//        rect.w = is->video_ctx->width;
//        rect.h = is->video_ctx->height; 修改的渲染的尺寸，因为原始尺寸是视频帧中的大小，可能很大
        rect.w = PLAYER_WIDTH;
        rect.h = PLAYER_HEIGHT;
        SDL_LockMutex(text_mutex);
        SDL_RenderClear( renderer );//刷一次屏
        SDL_RenderCopy( renderer, texture, NULL, &rect);//将数据拷贝到renderer中去
        SDL_RenderPresent( renderer );//展示渲染器中的内容
        SDL_UnlockMutex(text_mutex);
        
    }
}

void video_refresh_timer(void *userdata) {
    
    VideoState *is = (VideoState *)userdata;
    VideoPicture *vp;
    double actual_delay, delay, sync_threshold, ref_clock, diff;
    
    if(is->video_st) {
        if(is->pictq_size == 0) {//解码后的队列是否有数据
            schedule_refresh(is, 1);//每1毫秒刷新一次
            //fprintf(stderr, "no picture in the queue!!!\n");
        } else {
            //fprintf(stderr, "get picture from queue!!!\n");
            vp = &is->pictq[is->pictq_rindex];//解码后的视频帧
            
            is->video_current_pts = vp->pts;
            is->video_current_pts_time = av_gettime();
//          printf("is->video_current_pts_time = %lld --- \n",is->video_current_pts_time);//竟然是负值
           
            delay = vp->pts - is->frame_last_pts; /* the pts from last time */
#if DEBUG1
            printf("vp->pts = %lf ---\n",vp->pts);
            printf("is->frame_last_pts = %lf ---\n",is->frame_last_pts);
            printf("delay = %lf ---\n",delay);
#endif
            if(delay <= 0 || delay >= 1.0) { //时间单位是秒
                /* if incorrect delay, use previous one */
                delay = is->frame_last_delay;
            }
            /* save for next time */
            is->frame_last_delay = delay;//更新frame_last_delay 方便下一次进行计算
            is->frame_last_pts = vp->pts;//
            
            /* update delay to sync to audio if not master source */
            if(is->av_sync_type != AV_SYNC_VIDEO_MASTER) {
                ref_clock = get_audio_clock(is);//拿到参考时钟
                diff = vp->pts - ref_clock; //视频帧的pts 减去 音频参考帧的pts
#if DEBUG1
                printf("diff = %lf --\n",diff);
                /*
                 时间基
                 tbr:帧率
                 tbn:time base of stream
                 tbc:time base of codec
                 
                 计算当前帧的PTS
                 PTS = PTS * av_q2d(video_stream->time_base)
                 
                 计算下一帧的PTS
                 video_clock:预测的下一帧视频的PTS
                 frame_delay:1/tbr
                 
                 audio_clock:音频当前播放的时间戳
                 */
                /*
                 视频播放的基本思路
                 
                 一般的做法、展示第一帧视频后，获得要显示的下一个视频帧PTS,然后设置一个定时器，当定时器超时后，刷新新的视频帧如此反复操作.
                 
                 视音频都通过回调的形式，循环解码、播放
                 
                 */
#endif
                /* Skip or repeat the frame. Take delay into account
                 FFPlay still doesn't "know if this is the best guess." */
                sync_threshold = (delay > AV_SYNC_THRESHOLD) ? delay : AV_SYNC_THRESHOLD;//更新阈值
                /*
                 AV_SYNC_THRESHOLD 最小的阈值10毫秒，是根据音频最小就是10毫秒
                 */
                
                /*
                 delay是上一个视频帧的pts 减去当前帧的pts得到的差值 可以通过FLV视频格式分析器 查看Tag Header中的参数TimeStamp
                 */
#if DEBUG1
                printf("sync_threshold = %lf ---\n",sync_threshold);
                
                //比较的是  当前视频帧pts和音频帧pts之间的差值 diff ,当前视频帧pts和上一视频帧pts的 的时间差值 sync_threshold ，
#endif
                //如果diff的绝对值小于10秒，可以进行同步
                if(fabs(diff) < AV_NOSYNC_THRESHOLD) { //double fabs(double x) 返回 x 的绝对值
                    if(diff <= -sync_threshold) { //说明视频帧 在音频帧之前，视频需要加速播放
                        delay = 0;
                    } else if(diff >= sync_threshold) { //视频帧和音频帧的差值大约阈值，需要慢播
                        delay = 2 * delay;
                    }
                }
            }
            //系统时间加上delay
            is->frame_timer += delay;
            /* computer the REAL delay */
            actual_delay = is->frame_timer - (av_gettime() / 1000000.0);
            if(actual_delay < 0.010) {
                /* Really it should skip the picture instead */
                actual_delay = 0.010; // 这个相当于每秒播放100帧
            }
#if DEBUG1
            printf("actual_delay = %lf --  \n",actual_delay);
#endif
            schedule_refresh(is, (int)(actual_delay * 1000 + 0.5)); //每次都设置一个timer  10毫秒再增加0.5毫秒的微差值
            
            /* show the picture! */
            video_display(is); //展示当前帧
            
            /* update queue for next picture! */
            if(++is->pictq_rindex == VIDEO_PICTURE_QUEUE_SIZE) {
                is->pictq_rindex = 0;
            }
            SDL_LockMutex(is->pictq_mutex);
            is->pictq_size--;
            SDL_CondSignal(is->pictq_cond);//这个语句实际上是先解锁 在发的信号量 然后再加锁
            SDL_UnlockMutex(is->pictq_mutex);
        }
    } else {
        schedule_refresh(is, 100);
    }
}

void alloc_picture(void *userdata) {
    
    int ret;
    
    VideoState *is = (VideoState *)userdata;
    VideoPicture *vp;
    
    vp = &is->pictq[is->pictq_windex];
    if(vp->bmp) {
        
        // we already have one make another, bigger/smaller
        avpicture_free(vp->bmp);
        free(vp->bmp);
        
        vp->bmp = NULL;
    }
    
    // Allocate a place to put our YUV image on that screen
    SDL_LockMutex(text_mutex);
    
    vp->bmp = (AVPicture*)malloc(sizeof(AVPicture));
    ret = avpicture_alloc(vp->bmp, AV_PIX_FMT_YUV420P, is->video_ctx->width, is->video_ctx->height);
    if (ret < 0) {
        fprintf(stderr, "Could not allocate temporary picture: %s\n", av_err2str(ret));
    }
    
    SDL_UnlockMutex(text_mutex);
    
    vp->width = is->video_ctx->width ;
    vp->height = is->video_ctx->height ;
//    vp->width = 640 ;
//    vp->height = 480 ;
    vp->allocated = 1;
    
}

//将解码后的视频帧插入到解码后的视频帧队列中
int queue_picture(VideoState *is, AVFrame *pFrame, double pts) {
    
    VideoPicture *vp;
    
    /* wait until we have space for a new pic */
    SDL_LockMutex(is->pictq_mutex);
    while(is->pictq_size >= VIDEO_PICTURE_QUEUE_SIZE &&
          !is->quit) {
        SDL_CondWait(is->pictq_cond, is->pictq_mutex);
    }
    SDL_UnlockMutex(is->pictq_mutex);
    
    if(is->quit) //是否需要退出
        return -1;
    
    // windex is set to 0 initially
    vp = &is->pictq[is->pictq_windex];
    
    /* allocate or resize the buffer! */
    //picture是空的，或者宽高不匹配
    if(!vp->bmp ||
       vp->width != is->video_ctx->width ||
       vp->height != is->video_ctx->height) {
        
        vp->allocated = 0;
        alloc_picture(is);
        if(is->quit) {
            return -1;
        }
    }
    
    /* We have a place to put our picture on the queue */
    if(vp->bmp) {
        
        vp->pts = pts;
        
        // Convert the image into YUV format that SDL uses
        sws_scale(is->video_sws_ctx, (uint8_t const * const *)pFrame->data,
                  pFrame->linesize, 0, is->video_ctx->height ,
                  vp->bmp->data, vp->bmp->linesize);
        
        /* now we inform our display thread that we have a pic ready */
        if(++is->pictq_windex == VIDEO_PICTURE_QUEUE_SIZE) {
            is->pictq_windex = 0;
        }
        
        SDL_LockMutex(is->pictq_mutex);
        is->pictq_size++;
        SDL_UnlockMutex(is->pictq_mutex);
    }
    return 0;
}

double synchronize_video(VideoState *is, AVFrame *src_frame, double pts) {
    
    double frame_delay;
    
    if(pts != 0) {
        /* if we have pts, set video clock to it */
        is->video_clock = pts; //如果pts不为零 更新一下视频时钟
    } else {
        /* if we aren't given a pts, set it to the clock */
        pts = is->video_clock; //如果是pts是0，使用上一个pts
    }
    /* update the video clock */
    frame_delay = av_q2d(is->video_ctx->time_base); //通过时间基计算出 每一帧之间的时间间隔
    /* if we are repeating a frame, adjust clock accordingly */
    frame_delay += src_frame->repeat_pict * (frame_delay * 0.5); //解码后的视频帧，可能要重复的播放
    /**
        * When decoding, this signals how much the picture must be delayed.
        * extra_delay = repeat_pict / (2*fps)
        * int repeat_pict;
        */
#if DEBUG1
    printf("src_frame->repeat_pict = %d --\n",src_frame->repeat_pict);
#endif
    is->video_clock += frame_delay; //保存为下一帧的pts
    return pts;
}
//解码视频线程
/*
 在解码前的AVPacket中获取
 在解码后的AVFrame中获取
 通过函数av_frame_get_best_effort_timestamp(frame)函数获取
 */
int decode_video_thread(void *arg) {
    VideoState *is = (VideoState *)arg;
    AVPacket pkt1, *packet = &pkt1;
    int frameFinished;
    AVFrame *pFrame;
    double pts;
    
    pFrame = av_frame_alloc();
    
    for(;;) {
        if(packet_queue_get(&is->videoq, packet, 1) < 0) { //从视频流队列中取出视频包packet
            // means we quit getting packets
            break;
        }
        pts = 0;
        
        // Decode video frame
        avcodec_decode_video2(is->video_ctx, pFrame, &frameFinished, packet); //对packet进行解码 成功的标记放在frameFinished中，解码成功的数据放在pFrame中
        
        if((pts = av_frame_get_best_effort_timestamp(pFrame)) != AV_NOPTS_VALUE) {
        } else {
            pts = 0;
        }
        pts *= av_q2d(is->video_st->time_base); //可以根据pts来计算一帧在整个视频中的时间位置(有待分析)
#if DEBUG1
        printf("(pts = av_frame_get_best_effort_timestamp(pFrame) = %lf --\n",pts);
        printf("is->video_st->time_base.num = %d --\n",is->video_st->time_base.num);
        printf("is->video_st->time_base.den = %d --\n",is->video_st->time_base.den);
#endif
        /*
         计算视频长度的方法：
         
         time(秒) = st->duration * av_q2d(st->time_base)
         */
        
        // Did we get a video frame?
        if(frameFinished) {
            pts = synchronize_video(is, pFrame, pts);
            if(queue_picture(is, pFrame, pts) < 0) { //将解码后的视频帧放入队列中去
                break;
            }
        }
        av_free_packet(packet);
    }
    av_frame_free(&pFrame);
    return 0;
}

//流的组件打开
int stream_component_open(VideoState *is, int stream_index) {
    
    AVFormatContext *pFormatCtx = is->pFormatCtx;
    AVCodecContext *codecCtx = NULL;
    AVCodec *codec = NULL;
    SDL_AudioSpec wanted_spec, spec;
    
    if(stream_index < 0 || stream_index >= pFormatCtx->nb_streams) { //判断是否数组越界
        return -1;
    }
    
    codecCtx = avcodec_alloc_context3(NULL);//只创建不进行初始化
    
    
    int ret = avcodec_parameters_to_context(codecCtx, pFormatCtx->streams[stream_index]->codecpar);
    if (ret < 0)
        return -1;
    
    codec = avcodec_find_decoder(codecCtx->codec_id); //找到解码器
    if(!codec) {
        fprintf(stderr, "Unsupported codec!\n");
        return -1;
    }
    
    if(avcodec_open2(codecCtx, codec, NULL) < 0) { //打开解码器
        fprintf(stderr, "Unsupported codec!\n");
        return -1;
    }
    
    switch(codecCtx->codec_type) {
        case AVMEDIA_TYPE_AUDIO:
            
            // Set audio settings from codec info
            wanted_spec.freq = codecCtx->sample_rate; //采样率
            wanted_spec.format = AUDIO_S16SYS; //采样格式
            wanted_spec.channels = 2;//codecCtx->channels;
            wanted_spec.silence = 0;
            wanted_spec.samples = SDL_AUDIO_BUFFER_SIZE; //采样的个数
            wanted_spec.callback = audio_callback; //回调函数
            wanted_spec.userdata = is; //回调函数的参数
            
            fprintf(stderr, "wanted spec: channels:%d, sample_fmt:%d, sample_rate:%d \n",
                    2, AUDIO_S16SYS, codecCtx->sample_rate);
            
            if(SDL_OpenAudio(&wanted_spec, &spec) < 0) {  //打开音频设备
                fprintf(stderr, "SDL_OpenAudio: %s\n", SDL_GetError());
                return -1;
            }
            is->audio_hw_buf_size = spec.size;
            
            is->audioStream = stream_index; //音频流的index
            is->audio_st = pFormatCtx->streams[stream_index];//具体的音频流
            is->audio_ctx = codecCtx; //编解码器的上下文
            is->audio_buf_size = 0;//音频的buf_size
            is->audio_buf_index = 0;//使用了多少
            memset(&is->audio_pkt, 0, sizeof(is->audio_pkt));//设置音频包
            packet_queue_init(&is->audioq); //音频队列初始化
            
            //Out Audio Param
            uint64_t out_channel_layout=AV_CH_LAYOUT_STEREO;
            
            //AAC:1024  MP3:1152
            int out_nb_samples= is->audio_ctx->frame_size;
            //AVSampleFormat out_sample_fmt = AV_SAMPLE_FMT_S16;
            
            int out_sample_rate=is->audio_ctx->sample_rate;
            int out_channels=av_get_channel_layout_nb_channels(out_channel_layout);
            //Out Buffer Size
            /*
             int out_buffer_size=av_samples_get_buffer_size(NULL,
             out_channels,
             out_nb_samples,
             AV_SAMPLE_FMT_S16,
             1);
             */
            
            //uint8_t *out_buffer=(uint8_t *)av_malloc(MAX_AUDIO_FRAME_SIZE*2);
            int64_t in_channel_layout=av_get_default_channel_layout(is->audio_ctx->channels);
            
            struct SwrContext *audio_convert_ctx;  //音频重采样上下文
            audio_convert_ctx = swr_alloc();
            swr_alloc_set_opts(audio_convert_ctx,
                               out_channel_layout,//转换后的通道数
                               AV_SAMPLE_FMT_S16,//转换后的采样格式
                               out_sample_rate,//转换后的采样率
                               in_channel_layout,//转换前的的通道数
                               is->audio_ctx->sample_fmt, //转换前的采样格式
                               is->audio_ctx->sample_rate,//转换前的的采样率
                               0,
                               NULL);
            fprintf(stderr, "swr opts: out_channel_layout:%lld, out_sample_fmt:%d, out_sample_rate:%d, in_channel_layout:%lld, in_sample_fmt:%d, in_sample_rate:%d",
                    out_channel_layout, AV_SAMPLE_FMT_S16, out_sample_rate, in_channel_layout, is->audio_ctx->sample_fmt, is->audio_ctx->sample_rate);
            swr_init(audio_convert_ctx);//初始化
            
            is->audio_swr_ctx = audio_convert_ctx;
            
            SDL_PauseAudio(0);//播放音频
            break;
        case AVMEDIA_TYPE_VIDEO:
            is->videoStream = stream_index;//获取流的index
            is->video_st = pFormatCtx->streams[stream_index];//具体的流
            is->video_ctx = codecCtx;//视频解码上下文
            
            is->frame_timer = (double)av_gettime() / 1000000.0; //获取系统时间换算成秒
#if DEBUG1
            printf(" \n av_gettime() = %d -- \n",av_gettime());
            printf(" \n is->frame_timer = %lf -- \n",is->frame_timer);
#endif
            is->frame_last_delay = 40e-3;
            is->video_current_pts_time = av_gettime();
            
            packet_queue_init(&is->videoq);//创建视频队列
            
//            is->video_ctx->width  = PLAYER_WIDTH;
//            is->video_ctx->height = PLAYER_HEIGHT;
            
            printf("\n is->video_ctx->width = %d --\n", is->video_ctx->width);
            
            //创建视频裁剪的上下文
            is->video_sws_ctx = sws_getContext(is->video_ctx->width, is->video_ctx->height,
                                               is->video_ctx->pix_fmt,  is->video_ctx->width,
                                               is->video_ctx->height , AV_PIX_FMT_YUV420P,
                                               SWS_BILINEAR, NULL, NULL, NULL
                                               );
            
            is->video_tid = SDL_CreateThread(decode_video_thread, "decode_video_thread", is); //创建视频解码线程
            break;
        default:
            break;
    }
    //fix 1
    return 0;
}

//解复用线程
int demux_thread(void *arg) {
    
    int err_code;
    char errors[1024] = {0,};
    
    VideoState *is = (VideoState *)arg;
    AVFormatContext *pFormatCtx = NULL;
    AVPacket pkt1, *packet = &pkt1;
    
    int video_index = -1;
    int audio_index = -1;
    int i;
    
    is->videoStream=-1;
    is->audioStream=-1;
    
    global_video_state = is;
    
    /* open input file, and allocate format context */ //打开多媒体文件
    if ((err_code=avformat_open_input(&pFormatCtx, is->filename, NULL, NULL)) < 0) {
        av_strerror(err_code, errors, 1024);
        fprintf(stderr, "Could not open source file %s, %d(%s)\n", is->filename, err_code, errors);
        return -1;
    }
    
    is->pFormatCtx = pFormatCtx;
    
    // Retrieve stream information 检索流信息  都有哪些流
    if(avformat_find_stream_info(pFormatCtx, NULL)<0)
        return -1; // Couldn't find stream information
    
    // Dump information about file onto standard error
    av_dump_format(pFormatCtx, 0, is->filename, 0);
    
    // Find the first video stream  找到流的索引
    
    for(i=0; i<pFormatCtx->nb_streams; i++) {
        if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO &&
           video_index < 0) {
            video_index=i;
        }
        if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_AUDIO &&
           audio_index < 0) {
            audio_index=i;
        }
    }
    if(audio_index >= 0) {
        stream_component_open(is, audio_index);
    }
    if(video_index >= 0) {
        stream_component_open(is, video_index);
    }
    
    if(is->videoStream < 0 || is->audioStream < 0) { //判断音频流和视频流是否正常
        fprintf(stderr, "%s: could not open codecs\n", is->filename);
        goto fail;
    }
    
    screen_width = is->video_ctx->width ;
    screen_height = is->video_ctx->height; //屏幕的宽高等于 视频帧的宽高
    
//    screen_width = PLAYER_WIDTH ;
//    screen_height = PLAYER_HEIGHT; //屏幕的宽高等于 视频帧的宽高
    
    // main decode loop 主解码循环
    
    for(;;) { //从多媒体文件中不停的读取一个个的包
        if(is->quit) {
            break;
        }
        // seek stuff goes here 在这里找东西
        if(is->audioq.size > MAX_AUDIOQ_SIZE ||
           is->videoq.size > MAX_VIDEOQ_SIZE) {
            SDL_Delay(10);
            continue;
        }
        if(av_read_frame(is->pFormatCtx, packet) < 0) { //解复用
            if(is->pFormatCtx->pb->error == 0) {
                SDL_Delay(100); /* no error; wait for user input */
                continue;
            } else {
                break;
            }
        }
        // Is this a packet from the video stream?
        if(packet->stream_index == is->videoStream) { //如果是视频 就把包放到视频流队列
            packet_queue_put(&is->videoq, packet);
        } else if(packet->stream_index == is->audioStream) { //如果是音频就放到音频流队列
            packet_queue_put(&is->audioq, packet);
        } else {
            av_free_packet(packet);
        }
    }
    /* all done - wait for it */
    while(!is->quit) {
        SDL_Delay(100);
    }
    
fail:
    if(1){
        SDL_Event event;
        event.type = FF_QUIT_EVENT;
        event.user.data1 = is;
        
        SDL_PushEvent(&event);//功能全部完成之后 发送退出的信号
        printf("PUSH_QUIT_EVENT\n");
    }
    return 0;
}

int main(int argc, char *argv[]) {
    
    const char * path = "/Users/housenkui/Desktop/屏幕录制2020-04-09下午12.21.54.mov";
    
    SDL_Event       event;
    
    VideoState      *is;
    
    is = av_mallocz(sizeof(VideoState));
    
    audiofd = fopen("testout.pcm", "wb+");
    // Register all formats and codecs
    av_register_all();
    
    if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_TIMER)) {
        fprintf(stderr, "Could not initialize SDL - %s\n", SDL_GetError());
        exit(1);
    }
    
    //creat window from SDL
    win = SDL_CreateWindow("Media Player",
                           100,
                           100,
                           0,0,
//                           is->video_ctx->width, is->video_ctx->height,
                           SDL_WINDOW_RESIZABLE);
    if(!win) {
        fprintf(stderr, "\nSDL: could not set video mode:%s - exiting\n", SDL_GetError());
        exit(1);
    }
    
    renderer = SDL_CreateRenderer(win, -1, 0);
    
    text_mutex = SDL_CreateMutex();
    
    av_strlcpy(is->filename, path, sizeof(is->filename));
    
    is->pictq_mutex = SDL_CreateMutex(); //解码视频帧队列创建的锁
    is->pictq_cond = SDL_CreateCond();
    
    schedule_refresh(is, 40);//设置一个timer每40毫秒回调一次 去渲染视频帧
    
    is->av_sync_type = DEFAULT_AV_SYNC_TYPE;
    is->parse_tid = SDL_CreateThread(demux_thread,"demux_thread", is); //创建解复用线程
    if(!is->parse_tid) {
        av_free(is);
        return -1;
    }
    for(;;) {
        
        SDL_WaitEvent(&event);
        switch(event.type) {
            case FF_QUIT_EVENT:
            case SDL_QUIT:
                is->quit = 1;
                /*
                 1、主线程接收到退出事件
                 2.解复用线程在循环分流时对quit进行判断
                 3.视频解码线程从视频队列中取包时对quit进行判断
                 4.音频解码从音频流队列中取包时对quit进行判断
                 5.音频循环解码时对quit进行判断
                 6.在收到信号量消息时对quit进行判断
                 
                 */
                
                printf("SDL_Quit --Receive---\n");
                SDL_Quit();
                fclose(audiofd);
//                exit(0);//SDL_Quit(）在一些情况会无法关闭窗口
               
                return 0;
                break;
            case FF_REFRESH_EVENT:
                video_refresh_timer(event.user.data1);
                break;
            default:
                break;
        }
    }
    
    return 0;
    
}

