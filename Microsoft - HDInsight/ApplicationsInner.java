/**
 * Microsoft - Big Data Technology
 * https://github.com/Azure/azure-sdk-for-java/blob/master/hdinsight/resource-manager/v2018_06_01_preview/src/main/java/com/microsoft/azure/management/hdinsight/v2018_06_01_preview/implementation/ApplicationsInner.java
 *
 *  Created on: Mar 03, 2019
 *  Data Scientist: Tung Dang
 */

package com.microsoft.azure.management.hdinsight.v2018_06_01_preview.mirror;

import retrofit2.Retrofit;
import com.google.common.reflect.TypeToken;
import com.microsoft.azure.AzureServiceFuture;
import com.microsoft.azure.ListOperationCallback;
import com.microsoft.azure.management.hdinsight.v2018_06_01_preview.ErrorResponseException;
import com.microsoft.azure.Page;
import com.microsoft.azure.PagedList;
import com.microsoft.rest.ServiceCallback;
import com.microsoft.rest.ServiceFuture;
import com.microsoft.rest.ServiceResponse;
import com.microsoft.rest.Validator;
import java.io.IOException;
import java.util.List;
import okhttp3.ResponseBody;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Header;
import retrofit2.http.Headers;
import retrofit2.http.HTTP;
import retrofit2.http.Path;
import retrofit2.http.PUT;
import retrofit2.http.Query;
import retrofit2.http.Url;
import retrofit2.Response;
import rx.functions.Func1;
import rx.Observable;

/***
 * An instance of this class provides access to all the operations defined in Application
 */

public class ApplicationsInner 
{
    private ApplicationsService service;
    private HDInsightManagementClientImpl client;

    public ApplicationsInner(Retrofit retrofit, HDInsightManagementClientImpl client)
    {
        this.service = retrofit.create(ApplicationsService.class);
        this.client = client;
    }

    /**
     * Lists all of the applications for the HDInsight cluster
     */

    public PagedList<ApplicationsInner> listByCluster(final String resourceGroupName, final String clusterName)
    {
        ServiceResponse<Page<ApplicationsInner>> response = listByClusterSinglePageAsync(resourceGroupName, clusterName).toBlocking().single();
        return.new PagedList<ApplicationsInner>(response.body()) 
        {
            @Override
            public Page<ApplicationsInner> nextPage(String nextPageLink) 
            {
                return listByClusterSinglePageAsync(nextPageLink).toBlocking().single().body();
            }
        };
    }

    public ServiceFuture<List<ApplicationsInner>> listByClusterAsync(final String resourceGroupName, final String clusterName, 
                                                    final ListOperationCallback<ApplicationsInner> serviceCallback)
    {
        return AzureServiceFuture.fromPageResponse (
            listByClusterSinglePageAsync(resourceGroupName, clusterName),
            new Func1<String, Observable<ServiceResponse<Page<ApplicationsInner>>>>() {
                @Override
                public Observable<ServiceResponse<Page<ApplicationsInner>>> call(String nextPageLink)
                {
                    return listByClusterSinglePageAsync(nextPageLink);
                }
            },
            serviceCallback
        );
    }

    public Observable<Page<ApplicationsInner>> listByClusterAsync(final String resourceGroupName, final String clusterName)
    {
        return listByClusterWithServiceResponseAsync(resourceGroupName, clusterName)
            .map(new Func1<ServiceResponse<Page<ApplicationsInner>>, Page<ApplicationsInner>>() {
                @Override
                public Page<ApplicationsInner> call(ServiceResponse<Page<ApplicationsInner>> response) {
                    return response.body();
                }
            });
    }

    public Observable<ServiceResponse<Page<ApplicationsInner>>> listByClusterWithServiceResponseAsync(final String resourceGroupName,
                                                                    final String clusterName)
    {
        return listByClusterSinglePageAsync(resourceGroupName, clusterName) 
            .concatMap(new Func1<ServiceResponse<Page<ApplicationsInner>>, Observable<ServiceResponse<Page<ApplicationsInner>>>>() {
                @Override
                public Observable<ServiceResponse<Page<ApplicationsInner>>> call(ServiceResponse<Page<ApplicationsInner>> page) {
                    String nextPageLink = page.body().nextPageLink();
                    if (nextPageLink == null) 
                    {
                        return Observable.just(page);
                    }
                    return Observable.just(page).concatWith(listByClusterWithServiceResponseAsync(nextPageLink));
                }
            });
    }

    public Observable<ServiceResponse<Page<ApplicationsInner>>> listByClusterSinglePageAsync(final String resourceGroupName, 
                                                                    final String clusterName)
    {
        if (this.client.subscriptionId() == null) {
            throw new IllegalArgumentException("Parameter this.client.subscriptionId() is required and cannot be null.");
        }
        if (resourceGroupName == null) {
            throw new IllegalArgumentException("Parameter resourceGroupName is required and cannot be null.");
        }
        if (clusterName == null) {
            throw new IllegalArgumentException("Parameter clusterName is required and cannot be null.");
        }
        if (this.client.apiVersion() == null) {
            throw new IllegalArgumentException("Parameter this.client.apiVersion() is required and cannot be null.");
        }

        return service.listByCluster(this.client.subscriptionId(), resourceGroupName, clusterName, 
                                        this.client.acceptLanguage(), this.client.userAgent())
            .flatMap(new Func1<Response<ResponseBody>, Observable<ServiceResponse<Page<ApplicationsInner>>>>() {
                @Override
                public Observable<ServiceResponse<Page<ApplicationsInner>>> call(Response<ResponseBody> response) {
                    try {
                        ServiceResponse<PageImpl<ApplicationsInner>> result = listByClusterDelegate(response);
                        return Observable.just(new ServiceResponse<Page<ApplicationsInner>>(result.body(), result.response()));
                    } catch (Throwable t) {
                        return Observable.error(t);
                    }
                }
            });
    }

    private ServiceResponse<PageImpl<ApplicationsInner>> listByClusterDelegate(Response<ResponseBody> response) 
            throws ErrorResponseException, IOException, IllegalArgumentException {
        return this.client.restClient().responseBuilderFactory().<PageImpl<ApplicationsInner>, 
                    ErrorResponseException>newInstance(this.client.serializerAdapter()) 
                .register(200, new TypeToken<PageImpl<ApplicationsInner>>() {}.getType())
                .registerError(ErrorResponseException.class)
                .build(response);
    }

    
};
