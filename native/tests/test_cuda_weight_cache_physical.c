/* End-to-end native CUDA qualification; no external model files are needed.
 * --require-gpu makes an unavailable Driver/device a failure for remote runs. */
#define _POSIX_C_SOURCE 200809L
#include "public_api_internal.h"
#include "profiling.h"
#include "cJSON.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static VxReport report = VX_REPORT_INIT;
static char directory[512], graph_path[640], weight_path[640];
static int accepted_fd = -1;
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\nstatus=%d code=%d stage=%d backend=%s %s\n%s\n%s\n", \
    __FILE__,__LINE__,#x,report.status,report.code,report.stage,report.backend,report.message,report.route_evidence,report.fallback_evidence); exit(1); } } while(0)
static void cleanup(void) {
    if(accepted_fd>=0)close(accepted_fd);
    if(graph_path[0])unlink(graph_path);
    if(weight_path[0])unlink(weight_path);
    if(directory[0])rmdir(directory);
}
static int gpu_present(void) {
#if VOLVOXAI_ENABLE_CUDA
    void* library=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    if(!library)return 0;
    int (*init)(unsigned)=NULL;int (*count)(int*)=NULL;void* symbol=dlsym(library,"cuInit");
    memcpy(&init,&symbol,sizeof(init));symbol=dlsym(library,"cuDeviceGetCount");memcpy(&count,&symbol,sizeof(count));
    int devices=0,ok=init&&count&&init(0)==0&&count(&devices)==0&&devices>0;
    dlclose(library);return ok;
#else
    return 0;
#endif
}
static cJSON* shape(int generation,int width) {
    cJSON* value=cJSON_CreateArray();CHECK(value);
    if(generation)CHECK(cJSON_AddItemToArray(value,cJSON_CreateNumber(1)));
    CHECK(cJSON_AddItemToArray(value,cJSON_CreateString("R")));
    if(width)CHECK(cJSON_AddItemToArray(value,cJSON_CreateNumber(width)));
    return value;
}
static cJSON* descriptor(const char* dtype,cJSON* dimensions) {
    cJSON* value=cJSON_CreateObject();CHECK(value&&dimensions);
    CHECK(cJSON_AddStringToObject(value,"dtype",dtype));CHECK(cJSON_AddItemToObject(value,"shape",dimensions));return value;
}
static cJSON* node(cJSON* nodes,const char* id,const char* op,const char* inputs) {
    cJSON* value=cJSON_CreateObject();CHECK(value);
    CHECK(cJSON_AddStringToObject(value,"id",id)&&cJSON_AddStringToObject(value,"opType",op));
    CHECK(cJSON_AddItemToObject(value,"inputs",cJSON_Parse(inputs)));
    CHECK(cJSON_AddObjectToObject(value,"outputs")&&cJSON_AddObjectToObject(value,"params"));
    CHECK(cJSON_AddItemToArray(nodes,value));return value;
}
static void output(cJSON* n,const char* port,const char* name,const char* dtype,cJSON* dims) {
    cJSON* value=descriptor(dtype,dims);CHECK(cJSON_AddStringToObject(value,"tensor",name));
    CHECK(cJSON_AddItemToObject(cJSON_GetObjectItemCaseSensitive(n,"outputs"),port,value));
}
static cJSON* make_graph(int gated,int generation,int incompatible) {
    cJSON* graph=cJSON_Parse("{\"format\":\"volvox-graph/v1\",\"dimensions\":{\"E\":{\"min\":8,\"max\":8}},\"banks\":{},\"inputs\":{},\"nodes\":[],\"outputs\":[],\"execution\":{\"stateless\":{},\"entrypoints\":[]}}");CHECK(graph);
    cJSON* dimensions=cJSON_GetObjectItemCaseSensitive(graph,"dimensions");
    cJSON* range=cJSON_AddObjectToObject(dimensions,"R");CHECK(range);
    CHECK(cJSON_AddNumberToObject(range,"min",1)&&cJSON_AddNumberToObject(range,"max",generation?6:4));
    CHECK(cJSON_AddStringToObject(cJSON_GetObjectItemCaseSensitive(graph,"banks"),gated?"gate":"experts","E"));
    cJSON* inputs=cJSON_GetObjectItemCaseSensitive(graph,"inputs");
    if(generation) {
        CHECK(cJSON_AddItemToObject(inputs,"tokens",descriptor("int32",shape(1,0))));
        CHECK(cJSON_AddItemToObject(inputs,"keep",descriptor("int32",shape(1,0))));
        CHECK(cJSON_AddItemToObject(inputs,"positions",descriptor("int32",shape(0,0))));
    } else CHECK(cJSON_AddItemToObject(inputs,"x",descriptor("float32",shape(0,4))));
    cJSON* nodes=cJSON_GetObjectItemCaseSensitive(graph,"nodes"),*n;
    if(generation) {
        n=node(nodes,"embedding","Embedding","{\"input\":\"tokens\",\"weight\":\"embedding\"}");
        output(n,"out","x","float32",shape(1,4));
    }
    n=node(nodes,"router","MoERouter","{\"input\":\"x\",\"weight\":\"router\"}");
    output(n,"indices","ids","float32",shape(generation,2));
    output(n,"weights","routes","float32",shape(generation,2));
    cJSON* params=cJSON_GetObjectItemCaseSensitive(n,"params");
    CHECK(cJSON_AddNumberToObject(params,"num_experts",8)&&cJSON_AddNumberToObject(params,"top_k",2));
    CHECK(cJSON_AddBoolToObject(params,"normalize",1)&&cJSON_AddNumberToObject(params,"temperature",1));
    n=node(nodes,"experts",gated?"MoEGatedMLP":"MoELinear",gated?
        "{\"input\":\"x\",\"gate_weight\":\"gate\",\"up_weight\":\"up\",\"down_weight\":\"down\",\"route_indices\":\"ids\",\"route_weights\":\"routes\"}":
        "{\"input\":\"x\",\"expert_weight\":\"experts\",\"expert_bias\":\"bias\",\"route_indices\":\"ids\",\"route_weights\":\"routes\"}");
    output(n,"out","y","float32",shape(generation,3));
    if(gated) {
        params=cJSON_GetObjectItemCaseSensitive(n,"params");
        CHECK(cJSON_AddBoolToObject(params,"route_input_scale",0)&&cJSON_AddNumberToObject(params,"swiglu_limit",1.5));
    }
    cJSON* outputs=cJSON_GetObjectItemCaseSensitive(graph,"outputs");
    CHECK(cJSON_AddItemToArray(outputs,cJSON_CreateString("y"))&&cJSON_AddItemToArray(outputs,cJSON_CreateString("ids")));
    if(generation) {
        n=node(nodes,"select","ArgMax","{\"input\":\"y\"}");output(n,"out","selected","int32",shape(1,0));
        params=cJSON_GetObjectItemCaseSensitive(n,"params");
        CHECK(cJSON_AddNumberToObject(params,"axis",2)&&cJSON_AddBoolToObject(params,"keepdims",0)&&cJSON_AddNumberToObject(params,"select_last_index",0));
        CHECK(cJSON_AddItemToArray(outputs,cJSON_CreateString("selected")));
    }
    if(incompatible) {
        n=node(nodes,"other_consumer","Identity","{\"input\":\"experts\"}");
        output(n,"out","leak","float32",cJSON_Parse("[8,4,3]"));
        CHECK(cJSON_AddItemToArray(outputs,cJSON_CreateString("leak")));
    }
    cJSON* execution=cJSON_GetObjectItemCaseSensitive(graph,"execution");
    cJSON* entry=cJSON_CreateObject();CHECK(entry);
    CHECK(cJSON_AddStringToObject(entry,"name","evaluate")&&cJSON_AddNumberToObject(entry,"effect",1));
    CHECK(cJSON_AddItemToObject(entry,"allowed_ranges",cJSON_Parse("[1]")));
    CHECK(cJSON_AddItemToObject(entry,"inputs",cJSON_Parse(generation?"[\"tokens\",\"keep\",\"positions\"]":"[\"x\"]")));
    CHECK(cJSON_AddItemToObject(entry,"outputs",cJSON_Duplicate(outputs,1)));
    CHECK(cJSON_AddItemToArray(cJSON_GetObjectItemCaseSensitive(execution,"entrypoints"),entry));
    if(generation)CHECK(cJSON_AddItemToObject(execution,"generation",cJSON_Parse(
        "{\"initialize_entrypoint\":\"evaluate\",\"transition_entrypoint\":\"evaluate\",\"token_input\":\"tokens\",\"keep_input\":\"keep\",\"token_output\":\"selected\",\"logits_output\":\"y\",\"logits_axis\":2,\"id_domain_size\":3,\"output_name\":\"generated\",\"sequence_axis\":1,\"full_prefix\":{\"position_input\":\"positions\"}}")));
    return graph;
}

typedef struct {const char* name;int rank,shape[3];size_t count;float data[256];} Weight;
static Weight* add_weight(Weight* all,size_t* count,const char* name,int rank,int a,int b,int c) {
    Weight* w=&all[(*count)++];memset(w,0,sizeof(*w));w->name=name;w->rank=rank;w->shape[0]=a;w->shape[1]=b;w->shape[2]=c;
    w->count=(size_t)a*b*(rank==3?(size_t)c:1);CHECK(w->count<=256);
    for(size_t i=0;i<w->count;i++)w->data[i]=(float)((int)((i*5+*count)%13)-6)*.125f;
    return w;
}
static void write_fixture(int gated,int generation,int bf16,int incompatible) {
    cJSON* graph=make_graph(gated,generation,incompatible);char* json=cJSON_PrintUnformatted(graph);CHECK(json);
    FILE* file=fopen(graph_path,"wb");CHECK(file);size_t n=strlen(json);CHECK(fwrite(json,1,n,file)==n&&!fclose(file));
    cJSON_free(json);cJSON_Delete(graph);
    Weight weights[6];size_t count=0;
    Weight* router=add_weight(weights,&count,"router",2,4,8,0);memset(router->data,0,sizeof(router->data));
    for(int i=0;i<4;i++){router->data[i*8+i*2]=1;router->data[i*8+i*2+1]=-1;}
    if(gated){add_weight(weights,&count,"gate",3,8,4,5);add_weight(weights,&count,"up",3,8,4,5);add_weight(weights,&count,"down",3,8,5,3);}
    else{add_weight(weights,&count,"experts",3,8,4,3);add_weight(weights,&count,"bias",2,8,3,0);}
    if(generation){Weight* embedding=add_weight(weights,&count,"embedding",2,3,4,0);const float values[]={4,3,0,0,-4,-3,0,0,0,0,4,3};memcpy(embedding->data,values,sizeof(values));}
    cJSON* header=cJSON_CreateObject();CHECK(header);size_t offset=0;
    for(size_t i=0;i<count;i++) {
        Weight* w=&weights[i];cJSON* tensor=cJSON_AddObjectToObject(header,w->name),*dims=cJSON_AddArrayToObject(tensor,"shape");CHECK(tensor&&dims);
        CHECK(cJSON_AddStringToObject(tensor,"dtype",bf16?(bf16==1?"BF16":"F16"):"F32"));
        for(int d=0;d<w->rank;d++)CHECK(cJSON_AddItemToArray(dims,cJSON_CreateNumber(w->shape[d])));
        cJSON* offsets=cJSON_AddArrayToObject(tensor,"data_offsets");CHECK(offsets);
        CHECK(cJSON_AddItemToArray(offsets,cJSON_CreateNumber((double)offset)));offset+=w->count*(bf16?2u:4u);
        CHECK(cJSON_AddItemToArray(offsets,cJSON_CreateNumber((double)offset)));
    }
    json=cJSON_PrintUnformatted(header);CHECK(json);size_t header_bytes=(strlen(json)+7u)&~(size_t)7;
    file=fopen(weight_path,"wb");CHECK(file);
    for(unsigned i=0;i<8;i++)CHECK(fputc((int)((uint64_t)header_bytes>>(8*i))&255,file)!=EOF);
    n=strlen(json);CHECK(fwrite(json,1,n,file)==n);for(;n<header_bytes;n++)CHECK(fputc(' ',file)!=EOF);
    for(size_t i=0;i<count;i++)for(size_t j=0;j<weights[i].count;j++) {
        uint32_t bits;memcpy(&bits,&weights[i].data[j],4);
        if(bf16==1)bits>>=16;
        else if(bf16==2)bits=((bits>>16)&0x8000u)|((bits&0x7fffffffu)?
            ((((bits>>23)&255u)-112u)<<10)|((bits>>13)&1023u):0u);
        for(unsigned b=0;b<(bf16?2u:4u);b++)CHECK(fputc((int)(bits>>(b*8))&255,file)!=EOF);
    }
    CHECK(!fclose(file));cJSON_free(json);cJSON_Delete(header);
}
static uint64_t source_hash(void) {
    unsigned char bytes[4096];uint64_t hash=UINT64_C(1469598103934665603);CHECK(lseek(accepted_fd,0,SEEK_SET)==0);
    ssize_t n;while((n=read(accepted_fd,bytes,sizeof(bytes)))>0)for(ssize_t i=0;i<n;i++){hash^=bytes[i];hash*=UINT64_C(1099511628211);}CHECK(n==0);return hash;
}
static VxModel* load(VxRuntime* runtime) {
    const char* weights=weight_path;VxModelSource source=VX_MODEL_SOURCE_INIT;source.graph_path=graph_path;source.weight_paths=&weights;source.weight_path_count=1;
    VxModel* model=NULL;CHECK(vx_runtime_load_model(runtime,&source,&model,&report)==VX_STATUS_OK&&model);return model;
}
static VxStatus compile(VxModel* model,const char* backend,const VxWeightBankCache* cache,size_t count,VxCompiledModel** out) {
    VxBackendPolicy policy=VX_BACKEND_POLICY_INIT;policy.mode=VX_BACKEND_REQUIRE;policy.backends=&backend;policy.backend_count=1;policy.operator_fallback=VX_OPERATOR_FALLBACK_FORBID;
    return vx_model_compile_options_revision_cache(model,&policy,1,NULL,VX_PACKED_ARITHMETIC_F32,cache,count,out,&report);
}
static VxExecutionContext* context(VxCompiledModel* compiled,int generation) {
    VxExecutionConfig config={.context=VX_CONTEXT_OPTIONS_INIT,.autoregressive=generation,.max_steps=generation?3:0,.sampling=VX_SAMPLING_GREEDY};
    VxExecutionContext* result=NULL;CHECK(vx_compiled_model_create_execution_context(compiled,&config,&result,&report)==VX_STATUS_OK&&result);return result;
}
static uint64_t device_bytes(VxExecutionContext* context,uint64_t* keys,size_t* count) {
    VxMemoryAllocation items[1024];VxMemoryInventory inventory={.items=items,.capacity=1024};VxMemoryView view;
    CHECK(vx_context_memory_inventory(context,&inventory,&view)==VX_STATUS_OK&&!inventory.truncated);
    uint64_t bytes=0;*count=0;
    for(size_t i=0;i<inventory.count;i++)if(items[i].allocator==VX_MEMORY_CUDA_GRAPH&&items[i].capacity){bytes+=items[i].capacity;CHECK(*count<128);keys[(*count)++]=items[i].key;}
    return bytes;
}
static void rejected(VxModel* model,const char* backend,const VxWeightBankCache* cache,size_t count) {
    VxCompiledModel* compiled=NULL;VxStatus status=compile(model,backend,cache,count,&compiled);
    CHECK(status!=VX_STATUS_OK&&!compiled&&report.stage==VX_STAGE_COMPILE);
}
static void near(float a,float b){CHECK(isfinite(a)&&isfinite(b)&&fabsf(a-b)<1e-4f*(1+fabsf(b)));}
static uint64_t compiled_domain_digest(void) {
    const char prefix[]="domain_digest=";const char* found=strstr(report.route_evidence,prefix);CHECK(found);
    const char* digits=found+sizeof(prefix)-1;char* end=NULL;uint64_t digest=strtoull(digits,&end,16);
    CHECK(end==digits+16);return digest;
}
static void execute(VxExecutionContext* context,const float* input,int rows,float* output_values,float* ids) {
    VxTensorBinding binding=VX_TENSOR_BINDING_INIT;binding.name="x";binding.dtype=VX_DTYPE_F32;binding.rank=2;binding.shape[0]=rows;binding.shape[1]=4;binding.data=input;binding.byte_size=(size_t)rows*4*sizeof(float);
    VxGraphExecutionOptions options={.entrypoint="evaluate",.range=VX_EXECUTION_RANGE_FULL};VxResult* result=NULL;size_t bytes=0;
    CHECK(vx_execution_context_execute_graph(context,&binding,1,&options,&result,&report)==VX_STATUS_OK&&result);
    CHECK(vx_result_read(result,"y",output_values,12*sizeof(float),&bytes,&report)==VX_STATUS_OK&&bytes==(size_t)rows*3*sizeof(float));
    CHECK(vx_result_read(result,"ids",ids,8*sizeof(float),&bytes,&report)==VX_STATUS_OK&&bytes==(size_t)rows*2*sizeof(float));vx_result_release(result);
}
static void graph_case(VxRuntime* runtime,int gated,int bf16) {
    write_fixture(gated,0,bf16,0);accepted_fd=open(weight_path,O_RDONLY);CHECK(accepted_fd>=0);uint64_t hash=source_hash();
    VxModel* model=load(runtime);CHECK(unlink(graph_path)==0&&unlink(weight_path)==0);
    VxWeightBankCache cache={gated?"gate":"experts",2};VxCompiledModel *cpu=NULL,*full=NULL,*cached=NULL;
    CHECK(compile(model,"cpu",NULL,0,&cpu)==VX_STATUS_OK);CHECK(compile(model,"cuda",NULL,0,&full)==VX_STATUS_OK);
    uint64_t full_digest=compiled_domain_digest();CHECK(compile(model,"cuda",&cache,1,&cached)==VX_STATUS_OK);
    CHECK(compiled_domain_digest()!=full_digest);
    VxCompiledDomainAttestationView ca={0},fa={0};CHECK(vx_compiled_model_internal_domain_attestation(cached,&ca)==VX_STATUS_OK&&vx_compiled_model_internal_domain_attestation(full,&fa)==VX_STATUS_OK);
    CHECK(ca.shape_domain_proof_identity&&ca.shape_domain_proof_identity[0]&&ca.maximum_tensor_bytes&&ca.maximum_resident_bytes);
    CHECK(!strcmp(ca.shape_domain_proof_identity,fa.shape_domain_proof_identity));uint64_t bound=ca.maximum_resident_bytes;
    VxWeightBankCache invalid={"missing",2};rejected(model,"cuda",&invalid,1);invalid.bank=cache.bank;invalid.resident_slots=0;rejected(model,"cuda",&invalid,1);invalid.resident_slots=9;rejected(model,"cuda",&invalid,1);invalid.resident_slots=1;rejected(model,"cuda",&invalid,1);
    VxWeightBankCache duplicate[]={cache,cache};rejected(model,"cuda",duplicate,2);rejected(model,"cpu",&cache,1);
    vx_model_release(model);
    VxExecutionContext* contexts[]={context(cpu,0),context(full,0),context(cached,0),context(cached,0)};
    vx_compiled_model_release(cpu);vx_compiled_model_release(full);vx_compiled_model_release(cached);
    const float seeds[]={4,3,0,0,-4,-3,0,0,0,0,4,3,0,0,-4,-3};
    const int sizes[]={4,1,3,2,4,4};uint64_t previous=0;
    for(size_t pass=0;pass<sizeof(sizes)/sizeof(sizes[0]);pass++) {
        int rows=sizes[pass];float input[16],reference[12],actual[12],reference_ids[8],ids[8];
        for(int r=0;r<rows;r++)for(int d=0;d<4;d++)input[r*4+d]=seeds[((r+(int)pass)%4)*4+d]*(1+.125f*(float)pass);
        execute(contexts[0],input,rows,reference,reference_ids);
        unsigned seen=0;for(int i=0;i<rows*2;i++)seen|=1u<<(unsigned)reference_ids[i];
        if(rows==4)CHECK(seen==255u);
        for(int c=1;c<4;c++){execute(contexts[c],input,rows,actual,ids);for(int i=0;i<rows*3;i++)near(actual[i],reference[i]);CHECK(!memcmp(ids,reference_ids,(size_t)rows*2*sizeof(float)));}
        uint64_t keys[128];size_t count;uint64_t used=device_bytes(contexts[2],keys,&count);CHECK(used&&used<=bound);
        if(pass)CHECK(used==previous);
        previous=used;
    }
    uint64_t full_keys[128],first[128],second[128];size_t full_count,a_count,b_count;
    uint64_t full_bytes=device_bytes(contexts[1],full_keys,&full_count),cached_bytes=device_bytes(contexts[2],first,&a_count);
    CHECK(device_bytes(contexts[3],second,&b_count)==cached_bytes&&cached_bytes<full_bytes&&full_count&&a_count&&b_count);
    static uint64_t f32_full_bytes[2],f32_cached_bytes[2];
    if(!bf16){f32_full_bytes[gated]=full_bytes;f32_cached_bytes[gated]=cached_bytes;}
    else CHECK(f32_full_bytes[gated]>full_bytes&&f32_cached_bytes[gated]>cached_bytes);
    for(size_t i=0;i<a_count;i++)for(size_t j=0;j<b_count;j++)CHECK(first[i]!=second[j]);
    CHECK(source_hash()==hash);
    printf("physical-cache %s/%s: full_device=%llu cached_device=%llu bound=%llu; compact storage, eviction, lifetime, isolation and parity PASS\n",gated?"gated":"linear",bf16?(bf16==1?"BF16":"F16"):"F32",(unsigned long long)full_bytes,(unsigned long long)cached_bytes,(unsigned long long)bound);
    for(int c=0;c<4;c++)vx_execution_context_release(contexts[c]);
    CHECK(close(accepted_fd)==0);accepted_fd=-1;
}
static void generation_case(VxRuntime* runtime,int gated,int cpu_only) {
    write_fixture(gated,1,0,0);VxModel* model=load(runtime);CHECK(!unlink(graph_path)&&!unlink(weight_path));
    VxCompiledModel* compiled[3]={NULL};VxWeightBankCache cache={gated?"gate":"experts",2};
    CHECK(compile(model,"cpu",NULL,0,&compiled[0])==VX_STATUS_OK);
    if(!cpu_only)CHECK(compile(model,"cuda",NULL,0,&compiled[1])==VX_STATUS_OK&&compile(model,"cuda",&cache,1,&compiled[2])==VX_STATUS_OK);
    vx_model_release(model);
    int32_t tokens[]={0,1},keep[]={1,1},positions[]={0,1},expected[3]={0};
    VxTensorBinding inputs[3]={VX_TENSOR_BINDING_INIT,VX_TENSOR_BINDING_INIT,VX_TENSOR_BINDING_INIT};
    for(int i=0;i<2;i++){inputs[i].name=i?"keep":"tokens";inputs[i].dtype=VX_DTYPE_I32;inputs[i].rank=2;inputs[i].shape[0]=1;inputs[i].shape[1]=2;inputs[i].data=i?(void*)keep:(void*)tokens;inputs[i].byte_size=8;}
    inputs[2].name="positions";inputs[2].dtype=VX_DTYPE_I32;inputs[2].rank=1;inputs[2].shape[0]=2;inputs[2].data=positions;inputs[2].byte_size=8;
    for(int c=0;c<(cpu_only?1:3);c++) {
        VxExecutionContext* instance=context(compiled[c],1);vx_compiled_model_release(compiled[c]);VxResult* result=NULL;
        VxGenerationOptions options={.initial_inputs_present=1,.max_steps=3};size_t bytes=0;int32_t actual[3];
        CHECK(vx_execution_context_generate(instance,inputs,3,&options,&result,&report)==VX_STATUS_OK&&result);
        CHECK(vx_result_read(result,"generated",actual,sizeof(actual),&bytes,&report)==VX_STATUS_OK&&bytes==sizeof(actual));
        if(!c)memcpy(expected,actual,sizeof(expected));else CHECK(!memcmp(expected,actual,sizeof(expected)));
        vx_result_release(result);vx_execution_context_release(instance);
    }
    printf("%s %s full-prefix Generate %s PASS\n",cpu_only?"fixture-cpu":"physical-cache",gated?"gated":"linear",cpu_only?"contract":"parity");
}
int main(int argc,char** argv) {
    int cpu_only=argc>1&&!strcmp(argv[1],"--cpu-generation");
    if(!cpu_only&&!gpu_present()){fprintf(stderr,"CUDA device unavailable\n");return argc>1&&!strcmp(argv[1],"--require-gpu")?1:77;}
    CHECK(setvbuf(stdout,NULL,_IOLBF,0)==0);
    const char* temporary=getenv("TMPDIR");if(!temporary)temporary="/tmp";
    CHECK(snprintf(directory,sizeof(directory),"%s/volvoxai-cache-physical-XXXXXX",temporary)>0&&mkdtemp(directory));
    CHECK(atexit(cleanup)==0);CHECK(snprintf(graph_path,sizeof(graph_path),"%s/model.graph.json",directory)>0&&snprintf(weight_path,sizeof(weight_path),"%s/weights.safetensors",directory)>0);
    VxRuntimeOptions options=VX_RUNTIME_OPTIONS_INIT;options.cpu_threads=1;options.execution_mode=VX_EXECUTION_MODE_DIRECT;VxRuntime* runtime=NULL;
    CHECK(vx_runtime_create(&options,&runtime,&report)==VX_STATUS_OK);
    if(!cpu_only)for(int gated=0;gated<2;gated++)for(int bf16=0;bf16<3;bf16++)graph_case(runtime,gated,bf16);
    for(int gated=0;gated<2;gated++)generation_case(runtime,gated,cpu_only);
    if(!cpu_only){write_fixture(0,0,0,1);VxModel* invalid=load(runtime);VxWeightBankCache cache={"experts",2};rejected(invalid,"cuda",&cache,1);vx_model_release(invalid);}
    vx_runtime_release(runtime);puts(cpu_only?"CPU generation fixture qualification PASS":"Physical CUDA weight-bank cache qualification PASS");return 0;
}
