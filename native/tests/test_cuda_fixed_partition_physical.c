/* Explicit CPU/CUDA placement qualification; all models are tiny owned files. */
#define _POSIX_C_SOURCE 200809L
#include "public_api_internal.h"
#include "profiling.h"
#include "safetensors.h"
#include "cJSON.h"
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 17
#define MAX_ROWS 6
static VxReport report = VX_REPORT_INIT;
static char directory[512], graph_path[640], weight_path[640];
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\nstatus=%d code=%d stage=%d %s\n%s\n%s\n", \
    __FILE__,__LINE__,#x,report.status,report.code,report.stage,report.message,report.route_evidence,report.fallback_evidence); exit(1); } } while(0)
static void cleanup(void) {
    if(graph_path[0])unlink(graph_path);
    if(weight_path[0])unlink(weight_path);
    if(directory[0])rmdir(directory);
}
static int gpu_present(void) {
#if VOLVOXAI_ENABLE_CUDA
    void* lib=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);if(!lib)return 0;
    int (*init)(unsigned)=NULL;int (*count)(int*)=NULL;void* symbol=dlsym(lib,"cuInit");
    memcpy(&init,&symbol,sizeof(init));symbol=dlsym(lib,"cuDeviceGetCount");memcpy(&count,&symbol,sizeof(count));
    int devices=0,ok=init&&count&&init(0)==0&&count(&devices)==0&&devices>0;dlclose(lib);return ok;
#else
    return 0;
#endif
}
static cJSON* dimensions(int generation,int width) {
    cJSON* result=cJSON_CreateArray();CHECK(result);
    if(generation)CHECK(cJSON_AddItemToArray(result,cJSON_CreateNumber(1)));
    CHECK(cJSON_AddItemToArray(result,cJSON_CreateString("R")));
    if(width)CHECK(cJSON_AddItemToArray(result,cJSON_CreateNumber(width)));
    return result;
}
static cJSON* descriptor(const char* dtype,cJSON* shape) {
    cJSON* result=cJSON_CreateObject();CHECK(result);
    CHECK(cJSON_AddStringToObject(result,"dtype",dtype)&&cJSON_AddItemToObject(result,"shape",shape));return result;
}
static cJSON* add_node(cJSON* nodes,const char* id,const char* op,cJSON* inputs,const char* output,const char* dtype,cJSON* shape) {
    cJSON* n=cJSON_CreateObject();CHECK(n);
    CHECK(cJSON_AddStringToObject(n,"id",id)&&cJSON_AddStringToObject(n,"opType",op));
    CHECK(cJSON_AddItemToObject(n,"inputs",inputs));
    cJSON* outs=cJSON_AddObjectToObject(n,"outputs"),*out=descriptor(dtype,shape);CHECK(outs);
    CHECK(cJSON_AddStringToObject(out,"tensor",output)&&cJSON_AddItemToObject(outs,"out",out));
    CHECK(cJSON_AddObjectToObject(n,"params")&&cJSON_AddItemToArray(nodes,n));return n;
}
static cJSON* graph(int generation,int shared) {
    cJSON* g=cJSON_Parse("{\"format\":\"volvox-graph/v1\",\"dimensions\":{\"R\":{\"min\":1,\"max\":6}},\"inputs\":{},\"nodes\":[],\"outputs\":[\"y\"],\"execution\":{\"stateless\":{},\"entrypoints\":[]}}");CHECK(g);
    cJSON* inputs=cJSON_GetObjectItemCaseSensitive(g,"inputs"),*nodes=cJSON_GetObjectItemCaseSensitive(g,"nodes");
    if(generation) {
        CHECK(cJSON_AddItemToObject(inputs,"tokens",descriptor("int32",dimensions(1,0))));
        CHECK(cJSON_AddItemToObject(inputs,"keep",descriptor("int32",dimensions(1,0))));
        CHECK(cJSON_AddItemToObject(inputs,"positions",descriptor("int32",dimensions(0,0))));
        add_node(nodes,"embedding","Embedding",cJSON_Parse("{\"input\":\"tokens\",\"weight\":\"embedding\"}"),"x","float32",dimensions(1,WIDTH));
    } else CHECK(cJSON_AddItemToObject(inputs,"x",descriptor("float32",dimensions(0,WIDTH))));
    const char* from[]={"x","a","b"};const char* to[]={"a","b","y"};
    for(int i=0;i<3;i++) {
        char id[16],weight[16];snprintf(id,sizeof(id),"dense%d",i);snprintf(weight,sizeof(weight),"w%d",shared&&i==1?0:i);
        cJSON* in=cJSON_CreateObject();CHECK(in&&cJSON_AddStringToObject(in,"input",from[i])&&cJSON_AddStringToObject(in,"weight",weight));
        cJSON* n=add_node(nodes,id,"Linear",in,to[i],"float32",dimensions(generation,WIDTH));
        CHECK(cJSON_AddStringToObject(cJSON_GetObjectItemCaseSensitive(n,"params"),"weight_layout",i==1?"din_dout":"dout_din"));
    }
    cJSON* outputs=cJSON_GetObjectItemCaseSensitive(g,"outputs");
    if(generation) {
        cJSON* n=add_node(nodes,"argmax","ArgMax",cJSON_Parse("{\"input\":\"y\"}"),"selected","int32",dimensions(1,0));
        cJSON* params=cJSON_GetObjectItemCaseSensitive(n,"params");CHECK(cJSON_AddNumberToObject(params,"axis",2)&&cJSON_AddBoolToObject(params,"keepdims",0)&&cJSON_AddNumberToObject(params,"select_last_index",0));
        CHECK(cJSON_AddItemToArray(outputs,cJSON_CreateString("selected")));
    }
    cJSON* execution=cJSON_GetObjectItemCaseSensitive(g,"execution"),*entry=cJSON_CreateObject();CHECK(entry);
    CHECK(cJSON_AddStringToObject(entry,"name","evaluate")&&cJSON_AddNumberToObject(entry,"effect",1));
    CHECK(cJSON_AddItemToObject(entry,"allowed_ranges",cJSON_Parse("[1]")));
    CHECK(cJSON_AddItemToObject(entry,"inputs",cJSON_Parse(generation?"[\"tokens\",\"keep\",\"positions\"]":"[\"x\"]")));
    CHECK(cJSON_AddItemToObject(entry,"outputs",cJSON_Duplicate(outputs,1))&&cJSON_AddItemToArray(cJSON_GetObjectItemCaseSensitive(execution,"entrypoints"),entry));
    if(generation)CHECK(cJSON_AddItemToObject(execution,"generation",cJSON_Parse(
        "{\"initialize_entrypoint\":\"evaluate\",\"transition_entrypoint\":\"evaluate\",\"token_input\":\"tokens\",\"keep_input\":\"keep\",\"token_output\":\"selected\",\"logits_output\":\"y\",\"logits_axis\":2,\"id_domain_size\":17,\"output_name\":\"generated\",\"sequence_axis\":1,\"full_prefix\":{\"position_input\":\"positions\"}}")));
    return g;
}
static float weight_value(int weight,int index) {return (float)((index*5+weight*3)%15-7)*.03125f;}
static void write_graph(cJSON* g) {
    char* text=cJSON_PrintUnformatted(g);CHECK(text);FILE* f=fopen(graph_path,"wb");CHECK(f);
    size_t bytes=strlen(text);CHECK(fwrite(text,1,bytes,f)==bytes&&!fclose(f));cJSON_free(text);
}
static void fixture(VxDataType dtype,int generation,int shared) {
    cJSON* g=graph(generation,shared);write_graph(g);cJSON_Delete(g);
    SafetensorsFile weights={0};CHECK(!safetensors_init_empty(&weights,SAFETENSORS_OPEN_READ_WRITE));
    const int shape[]={WIDTH,WIDTH};float values[WIDTH*WIDTH];uint16_t compact[WIDTH*WIDTH];
    for(int w=0;w<(generation?4:3);w++) {
        if(shared&&w==1)continue;
        char name[20];snprintf(name,sizeof(name),w==3?"embedding":"w%d",w);
        for(int i=0;i<WIDTH*WIDTH;i++) {
            values[i]=weight_value(w,i);uint32_t bits;memcpy(&bits,&values[i],sizeof(bits));
            compact[i]=dtype==VX_DTYPE_BF16?(uint16_t)(bits>>16):
                values[i]==0?0:(uint16_t)(((bits>>16)&0x8000u)|((((bits>>23)&255u)-127u+15u)<<10)|((bits>>13)&1023u));
        }
        CHECK(!safetensors_add_tensor(&weights,name,dtype,shape,2,dtype==VX_DTYPE_F32?(void*)values:(void*)compact,
            sizeof(values)/(dtype==VX_DTYPE_F32?1u:2u)));
    }
    CHECK(!safetensors_save(weight_path,&weights));safetensors_free(&weights);
}
static VxModel* load(VxRuntime* runtime) {
    const char* path=weight_path;VxModelSource source=VX_MODEL_SOURCE_INIT;source.graph_path=graph_path;source.weight_paths=&path;source.weight_path_count=1;
    VxModel* model=NULL;CHECK(vx_runtime_load_model(runtime,&source,&model,&report)==VX_STATUS_OK&&model);return model;
}
static VxStatus compile(VxModel* model,const char* backend,const char* const* cpu,size_t count,int allow_fallback,VxCompiledModel** result) {
    VxBackendPolicy policy=VX_BACKEND_POLICY_INIT;policy.mode=VX_BACKEND_REQUIRE;policy.backends=&backend;policy.backend_count=1;
    policy.operator_fallback=allow_fallback?VX_OPERATOR_FALLBACK_ALLOW:VX_OPERATOR_FALLBACK_FORBID;
    return vx_model_compile_options_revision_placement(model,&policy,0,NULL,VX_PACKED_ARITHMETIC_F32,NULL,0,cpu,count,result,&report);
}
static VxExecutionContext* context(VxCompiledModel* compiled,int generation) {
    VxExecutionConfig config={.context=VX_CONTEXT_OPTIONS_INIT,.autoregressive=generation,.max_steps=generation?3:0,.sampling=VX_SAMPLING_GREEDY};
    VxExecutionContext* result=NULL;CHECK(vx_compiled_model_create_execution_context(compiled,&config,&result,&report)==VX_STATUS_OK&&result);return result;
}
static void reject(VxModel* model,const char* backend,const char* const* cpu,size_t count,int allow_fallback) {
    VxCompiledModel* result=NULL;CHECK(compile(model,backend,cpu,count,allow_fallback,&result)!=VX_STATUS_OK&&!result&&report.stage==VX_STAGE_COMPILE);
}
static void compare(const float* actual,const float* expected,size_t count) {
    for(size_t i=0;i<count;i++)CHECK(isfinite(actual[i])&&isfinite(expected[i])&&fabsf(actual[i]-expected[i])<1e-4f*(1+fabsf(expected[i])));
}
static void reference(const float* input,int rows,int shared,float* output) {
    float current[MAX_ROWS*WIDTH],next[MAX_ROWS*WIDTH];memcpy(current,input,(size_t)rows*WIDTH*4);
    for(int w=0;w<3;w++) {
        for(int r=0;r<rows;r++)for(int out=0;out<WIDTH;out++) {
            double sum=0;for(int in=0;in<WIDTH;in++)sum+=(double)current[r*WIDTH+in]*weight_value(shared&&w==1?0:w,w==1?in*WIDTH+out:out*WIDTH+in);
            next[r*WIDTH+out]=(float)sum;
        }
        memcpy(current,next,(size_t)rows*WIDTH*4);
    }
    memcpy(output,current,(size_t)rows*WIDTH*4);
}
static void execute(VxExecutionContext* context,const float* input,int rows,float* output) {
    VxTensorBinding binding=VX_TENSOR_BINDING_INIT;binding.name="x";binding.dtype=VX_DTYPE_F32;binding.rank=2;
    binding.shape[0]=rows;binding.shape[1]=WIDTH;binding.data=input;binding.byte_size=(size_t)rows*WIDTH*4;
    VxGraphExecutionOptions options={.entrypoint="evaluate",.range=VX_EXECUTION_RANGE_FULL};VxResult* result=NULL;size_t bytes=0;
    CHECK(vx_execution_context_execute_graph(context,&binding,1,&options,&result,&report)==VX_STATUS_OK&&result);
    CHECK(vx_result_read(result,"y",output,MAX_ROWS*WIDTH*4,&bytes,&report)==VX_STATUS_OK&&bytes==(size_t)rows*WIDTH*4);vx_result_release(result);
}
static uint64_t inventory(VxExecutionContext* context,uint64_t* keys,size_t* count) {
    VxMemoryAllocation items[2048];VxMemoryInventory inventory={.items=items,.capacity=2048};VxMemoryView view;
    CHECK(vx_context_memory_inventory(context,&inventory,&view)==VX_STATUS_OK&&!inventory.truncated);uint64_t device=0;*count=0;
    for(size_t i=0;i<inventory.count;i++) {
        CHECK(items[i].allocator!=VX_MEMORY_HOST_PACKED_WEIGHTS);
        if(items[i].allocator==VX_MEMORY_CUDA_GRAPH&&items[i].capacity) {CHECK(*count<128);keys[(*count)++]=items[i].key;device+=items[i].capacity;}
    }
    return device;
}
static void graph_case(VxRuntime* runtime,VxDataType dtype,int shared,int cpu_fixture) {
    fixture(dtype,0,shared);VxModel* model=load(runtime);CHECK(!unlink(graph_path)&&!unlink(weight_path));
    const char* middle[]={"dense1"};const char* prefix[]={"dense0","dense1"};const char* suffix[]={"dense1","dense2"};
    const char* const* placements[]={middle,prefix,suffix};const size_t counts[]={1,2,2};
    VxCompiledModel* compiled[5]={NULL};CHECK(compile(model,"cpu",NULL,0,0,&compiled[0])==VX_STATUS_OK);
    if(!cpu_fixture) {
        CHECK(compile(model,"cuda",NULL,0,0,&compiled[1])==VX_STATUS_OK);
        for(int p=0;p<3;p++) {
            CHECK(compile(model,"cuda",placements[p],counts[p],0,&compiled[p+2])==VX_STATUS_OK);
            char placed[32];snprintf(placed,sizeof(placed),"cpu_placed=%zu",counts[p]);
            CHECK(!report.operator_fallback_used&&strstr(report.route_evidence,placed));
        }
        const char* unknown[]={"missing"};const char* duplicate[]={"dense1","dense1"};
        reject(model,"cuda",unknown,1,0);reject(model,"cuda",duplicate,2,0);reject(model,"cpu",middle,1,0);reject(model,"cuda",middle,1,1);
    }
    vx_model_release(model);
    VxExecutionContext* contexts[6]={NULL};int n=cpu_fixture?1:5;
    for(int c=0;c<n;c++) {contexts[c]=context(compiled[c],0);if(c==2)contexts[5]=context(compiled[c],0);vx_compiled_model_release(compiled[c]);}
    const int sizes[]={1,6,2,4,6,1};uint64_t previous[4]={0};
    for(size_t pass=0;pass<sizeof(sizes)/sizeof(sizes[0]);pass++) {
        int rows=sizes[pass];float input[MAX_ROWS*WIDTH],expected[MAX_ROWS*WIDTH],actual[MAX_ROWS*WIDTH];
        for(int i=0;i<rows*WIDTH;i++)input[i]=(float)((i*7+(int)pass)%19-9)*.125f;
        reference(input,rows,shared,expected);
        for(int c=0;c<n;c++) {execute(contexts[c],input,rows,actual);compare(actual,expected,(size_t)rows*WIDTH);}
        if(!cpu_fixture) {
            execute(contexts[5],input,rows,actual);compare(actual,expected,(size_t)rows*WIDTH);
            for(int c=1;c<5;c++) {uint64_t keys[128];size_t count;uint64_t bytes=inventory(contexts[c],keys,&count);CHECK(bytes&&count);if(pass)CHECK(previous[c-1]==bytes);previous[c-1]=bytes;}
        }
    }
    if(!cpu_fixture) {
        uint64_t first[128],second[128];size_t an,bn;uint64_t a=inventory(contexts[2],first,&an),b=inventory(contexts[5],second,&bn);CHECK(a==b);
        for(size_t i=0;i<an;i++)for(size_t j=0;j<bn;j++)CHECK(first[i]!=second[j]);
        uint64_t weight_bytes=WIDTH*WIDTH*(dtype==VX_DTYPE_F32?4u:2u);
        CHECK(previous[0]-previous[1]==(shared?0u:weight_bytes));
        CHECK(previous[0]-previous[2]==(shared?weight_bytes:2u*weight_bytes));
        CHECK(previous[0]-previous[3]==(shared?weight_bytes:2u*weight_bytes));
        printf("fixed-partition %s shared=%d: full=%llu middle=%llu prefix=%llu suffix=%llu; width=17, both layouts, parity, compact storage, fixed allocations, isolation and lifetime PASS\n",safetensors_dtype_name(dtype),shared,(unsigned long long)previous[0],(unsigned long long)previous[1],(unsigned long long)previous[2],(unsigned long long)previous[3]);
        vx_execution_context_release(contexts[5]);
    } else printf("CPU fixture %s shared=%d ExecuteGraph PASS\n",safetensors_dtype_name(dtype),shared);
    for(int c=0;c<n;c++)vx_execution_context_release(contexts[c]);
}
static void generation_case(VxRuntime* runtime,VxDataType dtype,int cpu_fixture) {
    fixture(dtype,1,0);VxModel* model=load(runtime);CHECK(!unlink(graph_path)&&!unlink(weight_path));
    const char* placement[]={"dense1"};VxCompiledModel* compiled[3]={NULL};CHECK(compile(model,"cpu",NULL,0,0,&compiled[0])==VX_STATUS_OK);
    if(!cpu_fixture)CHECK(compile(model,"cuda",NULL,0,0,&compiled[1])==VX_STATUS_OK&&compile(model,"cuda",placement,1,0,&compiled[2])==VX_STATUS_OK);
    vx_model_release(model);int32_t tokens[]={0,1},keep[]={1,1},positions[]={0,1},expected[3]={0};
    VxTensorBinding inputs[3]={VX_TENSOR_BINDING_INIT,VX_TENSOR_BINDING_INIT,VX_TENSOR_BINDING_INIT};
    for(int i=0;i<3;i++){inputs[i].name=i==0?"tokens":i==1?"keep":"positions";inputs[i].dtype=VX_DTYPE_I32;inputs[i].rank=i==2?1:2;inputs[i].shape[0]=i==2?2:1;inputs[i].shape[1]=2;inputs[i].data=i==0?tokens:i==1?keep:positions;inputs[i].byte_size=8;}
    for(int c=0;c<(cpu_fixture?1:3);c++) {
        VxExecutionContext* instance=context(compiled[c],1);vx_compiled_model_release(compiled[c]);VxResult* result=NULL;
        VxGenerationOptions options={.initial_inputs_present=1,.max_steps=3};int32_t actual[3];size_t bytes=0;
        CHECK(vx_execution_context_generate(instance,inputs,3,&options,&result,&report)==VX_STATUS_OK&&result);
        CHECK(vx_result_read(result,"generated",actual,sizeof(actual),&bytes,&report)==VX_STATUS_OK&&bytes==sizeof(actual));
        if(!c)memcpy(expected,actual,sizeof(actual));else CHECK(!memcmp(expected,actual,sizeof(actual)));
        vx_result_release(result);vx_execution_context_release(instance);
    }
    printf("%s %s full-prefix Generate PASS\n",cpu_fixture?"CPU fixture":"fixed-partition",safetensors_dtype_name(dtype));
}
static void stateful_refusal(VxRuntime* runtime) {
    fixture(VX_DTYPE_F32,1,0);cJSON* g=graph(1,0),*execution=cJSON_GetObjectItemCaseSensitive(g,"execution");
    cJSON_DeleteItemFromObjectCaseSensitive(execution,"stateless");cJSON_DeleteItemFromObjectCaseSensitive(execution,"generation");
    CHECK(cJSON_AddItemToObject(execution,"tensor_feedback",cJSON_Parse("{\"position_input\":\"positions\",\"bindings\":[{\"input_name\":\"tokens\",\"output_name\":\"selected\"}]}")));
    write_graph(g);cJSON_Delete(g);VxModel* model=load(runtime);const char* placement[]={"dense1"};reject(model,"cuda",placement,1,0);vx_model_release(model);
}
int main(int argc,char** argv) {
    int cpu_fixture=argc>1&&!strcmp(argv[1],"--cpu-fixture");
    if(!cpu_fixture&&!gpu_present()){fputs("CUDA device unavailable\n",stderr);return argc>1&&!strcmp(argv[1],"--require-gpu")?1:77;}
    CHECK(!setvbuf(stdout,NULL,_IOLBF,0));const char* temporary=getenv("TMPDIR");if(!temporary)temporary="/tmp";
    CHECK(snprintf(directory,sizeof(directory),"%s/volvoxai-partition-XXXXXX",temporary)>0&&mkdtemp(directory));CHECK(!atexit(cleanup));
    CHECK(snprintf(graph_path,sizeof(graph_path),"%s/model.graph.json",directory)>0&&snprintf(weight_path,sizeof(weight_path),"%s/weights.safetensors",directory)>0);
    VxRuntimeOptions options=VX_RUNTIME_OPTIONS_INIT;options.cpu_threads=1;options.execution_mode=VX_EXECUTION_MODE_DIRECT;VxRuntime* runtime=NULL;
    CHECK(vx_runtime_create(&options,&runtime,&report)==VX_STATUS_OK);
    const VxDataType dtypes[]={VX_DTYPE_F32,VX_DTYPE_F16,VX_DTYPE_BF16};
    for(size_t d=0;d<3;d++){graph_case(runtime,dtypes[d],0,cpu_fixture);graph_case(runtime,dtypes[d],1,cpu_fixture);generation_case(runtime,dtypes[d],cpu_fixture);}
    if(!cpu_fixture)stateful_refusal(runtime);
    vx_runtime_release(runtime);puts(cpu_fixture?"CPU partition fixture qualification PASS":"Physical CUDA fixed partition qualification PASS");return 0;
}
