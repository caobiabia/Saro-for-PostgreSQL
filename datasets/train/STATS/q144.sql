select  count(*) from posts as p,  		postLinks as pl,          postHistory as ph where p.Id = pl.PostId 	and pl.PostId = ph.PostId  AND ph.CreationDate<='2014-09-05 07:59:17'::timestamp  AND pl.LinkTypeId=1;