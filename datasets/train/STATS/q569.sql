select  count(*) from comments as c,  		postHistory as ph,          votes as v,  		users as u where u.Id  = v.UserId  	and v.UserId = ph.UserId  	and ph.UserId =c.UserId  AND c.CreationDate<='2014-09-04 22:48:46'::timestamp  AND u.UpVotes<=17  AND v.VoteTypeId=2;