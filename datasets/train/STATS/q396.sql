select  count(*) from comments as c,  		posts as p,          postHistory as ph,          votes as v,          badges as b,          users as u  where u.Id = p.OwnerUserId     and u.Id = b.UserId     and p.Id = c.PostId     and p.Id = ph.PostId     and p.Id = v.PostId  AND b.Date>='2010-07-20 09:59:09'::timestamp  AND b.Date<='2014-08-31 05:29:23'::timestamp  AND p.FavoriteCount<=2  AND p.CreationDate>='2010-07-20 04:16:52'::timestamp  AND u.Reputation>=1  AND u.Views>=0  AND u.CreationDate<='2014-08-11 05:51:07'::timestamp;